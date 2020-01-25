import pkg_resources
import math
import eth_utils
import itertools
import hashlib
import logging

logger = logging.getLogger("util")

def get_version():
    return pkg_resources.get_distribution("eth-index").version

"""
Merkle tree node utilities
"""
TREE_HEIGHT = 32
TREE_WIDTH = 2 ** 32
NODE_HASH_LENGTH = 32
ZERO_HASH = "0x000000000000000000000000000000000000000000000000000000"

def get_initial_metadata(merkle_tree_address):
  return {
    "merkletreeaddress": merkle_tree_address,
    "treeheight": TREE_HEIGHT,
    "latestblocknumber": 0,
    "latestroot": ZERO_HASH,
    "latestleafindex": 0,
    "latestfrontier": [ZERO_HASH] * (TREE_HEIGHT + 1),
  }

def concatenate_then_hash(items):
  concat_value = "".join([
    eth_utils.remove_0x_prefix(item)
    for item in items
  ])
  hash_bytes = hashlib.sha256(eth_utils.to_bytes(hexstr=concat_value)).digest()
  return eth_utils.to_hex(hash_bytes)

def right_shift(integer, shift):
  return math.floor(integer / 2 ** shift)

def left_shift(integer, shift):
  return integer * 2 ** shift

def leaf_index_to_node_index(leaf_index, tree_width=TREE_WIDTH):
  return leaf_index + tree_width - 1

def node_index_to_leaf_index(node_index, tree_width=TREE_WIDTH):
  return node_index + 1 - tree_width

def parent_node_index(node_index):
  if node_index % 2 == 1:
    return right_shift(node_index, 1)
  return right_shift(node_index - 1, 1)

def get_number_of_hashes(max_leaf_index, min_leaf_index, height=TREE_HEIGHT):
  hash_count = 0
  hi = max_leaf_index
  lo = min_leaf_index
  batch_size = max_leaf_index - min_leaf_index + 1
  bin_hi = bin(hi)
  bit_length = len(bin_hi)

  for i in range(0, bit_length):
    increment = hi - lo
    hash_count += increment
    hi = right_shift(hi, 1)
    lo = right_shift(lo, 1)

  return hash_count + height - (batch_size - 1)

def get_frontier_slot(leaf_index):
  slot = 0
  if (leaf_index % 2 == 1):
    exp_1 = 1
    pow_1 = 2
    pow_2 = pow_1 << 1
    while slot == 0:
      if ((leaf_index + 1 - pow_1) % pow_2 == 0):
        slot = exp_1
      else:
        pow_1 = pow_2
        pow_2 <<= 1
        exp_1 += 1
  return slot

def update_nodes(
  leaf_values,
  current_leaf_count,
  frontier,
  update_node_cb
):
  new_frontier = frontier

  number_of_leaves_available = TREE_WIDTH - current_leaf_count
  number_of_leaves = min([len(leaf_values), number_of_leaves_available])

  for leaf_index in range(current_leaf_count, current_leaf_count + number_of_leaves):
    """
    The node value before truncation (truncation is sometimes done so that the nodeValue (when concatenated with another) fits into a single hashing round in the next hashing iteration up the tree).
    """
    node_value_full = leaf_values[leaf_index - current_leaf_count]
    """
    Truncate hashed value, so it 'fits' into the next hash.
    """
    node_value = eth_utils.add_0x_prefix(node_value_full[-NODE_HASH_LENGTH * 2:])
    node_index = leaf_index_to_node_index(leaf_index, TREE_WIDTH)
    slot = get_frontier_slot(leaf_index)

    if slot == 0:
      new_frontier[slot] = node_value

    for level in range(1, slot + 1):
      if node_index % 2 == 0:
        node_value_full = concatenate_then_hash([
          frontier[level - 1],
          node_value
        ])
      else:
        node_value_full = concatenate_then_hash([
          node_value,
          ZERO_HASH
        ])
        node_value = eth_utils.add_0x_prefix(node_value_full[-NODE_HASH_LENGTH * 2])
      
      node_index = parent_node_index(node_index)
      update_node_cb(node_index, node_value)

    new_frontier[slot] = node_value
  
  for level in range(slot + 1, TREE_HEIGHT + 1):
    if node_index % 2 == 0:
      node_value_full = concatenate_then_hash([
        frontier[level - 1],
        node_value
      ])
      node_value = eth_utils.add_0x_prefix(
        node_value_full[-NODE_HASH_LENGTH * 2:]
      )
    else:
      node_value_full = concatenate_then_hash([
        node_value,
        ZERO_HASH
      ])
      node_value = eth_utils.add_0x_prefix(
        node_value_full[-NODE_HASH_LENGTH * 2:]
      )
    node_index = parent_node_index(node_index)

    node = {
      "value": node_value,
      "node_index": node_index
    }
    if node_index == 0:
      node["value"] = node_value_full
    
    update_node_cb(node_index, node_value=node["value"])
  
  root = node_value_full
  return (root, new_frontier)

