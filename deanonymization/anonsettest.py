from api_clients.mempool_client import MempoolClient
import duckdb
from models import DatabaseConnection
import argparse
from typing import Dict, Set, Tuple, List, Any
from itertools import combinations


def setup_phase(tx):
    """
    SETUP PHASE: Create sets for easy membership checking and removal

    Args:
        tx: CoinjoinTransaction object

    Returns:
        Tuple of (all_input_addresses, all_output_addresses,
                  inputs_by_value_and_type, outputs_by_value_and_type)
    """
    # Create a set of all input addresses for quick lookup
    all_input_addresses = {inp.script_pubkey_address for inp in tx.inputs}

    # Create a set of all output addresses for quick lookup
    all_output_addresses = {out.script_pubkey_address for out in tx.outputs}

    # Create sets grouped by value and script type for inputs
    inputs_by_value_and_type = {}
    for inp in tx.inputs:
        key = (inp.input_value_satoshi, inp.script_pubkey_type)
        if key not in inputs_by_value_and_type:
            inputs_by_value_and_type[key] = set()
        inputs_by_value_and_type[key].add(inp.script_pubkey_address)

    # Create sets grouped by value and script type for outputs
    outputs_by_value_and_type = {}
    for out in tx.outputs:
        key = (out.output_value_satoshi, out.script_pubkey_type)
        if key not in outputs_by_value_and_type:
            outputs_by_value_and_type[key] = set()
        outputs_by_value_and_type[key].add(out.script_pubkey_address)

    return all_input_addresses, all_output_addresses, inputs_by_value_and_type, outputs_by_value_and_type


def first_pass_build_anonymity_sets(tx) -> Tuple[Dict, Set, int]:
    """
    FIRST PASS: Build anonymity sets from outputs and track script type counts

    Args:
        tx: CoinjoinTransaction object

    Returns:
        Tuple of (anonsets, avasets, output_count)
    """
    output_count = 0
    avasets = set()
    anonsets = {}

    for out in tx.outputs:
        if out.output_value_satoshi not in avasets:
            anonsets[str(out.output_value_satoshi)] = {
                "outset": {out.script_pubkey_address},
                "inset": set(),
                "set_state": out.script_pubkey_type,
                "anon_size": 1,
                "output_type_counts": {out.script_pubkey_type: 1},  # Track script types in outputs
                "input_type_counts": {},  # Track script types added as inputs
                "inputs_added_for_distinct_outputs": {}
            }
            avasets.add(out.output_value_satoshi)
            output_count += 1
        else:
            cur = anonsets[str(out.output_value_satoshi)]
            outset = cur["outset"]
            outset: set
            outset.add(out.script_pubkey_address)
            cur["anon_size"] += 1

            # Update script type count for this output
            if out.script_pubkey_type in cur["output_type_counts"]:
                cur["output_type_counts"][out.script_pubkey_type] += 1
            else:
                cur["output_type_counts"][out.script_pubkey_type] = 1

            output_count += 1
            if cur["set_state"] != out.script_pubkey_type:
                cur["set_state"] = "other"

    return anonsets, avasets, output_count


def second_pass_add_inputs_to_anonsets(
    inputs_by_value_and_type: Dict,
    avasets: Set,
    anonsets: Dict,
    all_input_addresses: Set
) -> int:
    """
    SECOND PASS: Add inputs to anonymity sets, respecting script type distribution

    Args:
        inputs_by_value_and_type: Dictionary mapping (value, script_type) -> set of addresses
        avasets: Set of available output values
        anonsets: Dictionary of anonymity sets
        all_input_addresses: Set of all input addresses (modified in-place)

    Returns:
        input_count: Number of inputs matched
    """
    input_count = 0

    # Iterate over the grouped input sets for efficiency
    for (value, script_type), input_addrs in inputs_by_value_and_type.items():
        if value in avasets:
            cur = anonsets[str(value)]
            outset = cur["outset"]
            inset = cur["inset"]
            output_type_counts = cur["output_type_counts"]
            input_type_counts = cur["input_type_counts"]
            input_distinct_count = cur["inputs_added_for_distinct_outputs"]

            # Get the count of this script type in outputs
            output_count_for_type = output_type_counts.get(script_type, 0)

            # Skip if this script type doesn't exist in outputs
            if output_count_for_type == 0:
                continue

            # Get the count of this script type already added as inputs
            input_count_for_type = input_type_counts.get(script_type, 0)
            input_count_distinct_output = input_distinct_count.get(script_type, 0)

            # Calculate how many more inputs of this type we can add
            slots_available = min(
                output_count_for_type - input_count_for_type,  # Type-specific slots
                len(outset) - len(inset)  # Total slots in anonymity set
            )

            if slots_available > 0:
                # Convert to list to allow removal during iteration
                input_addrs_list = list(input_addrs)

                for addr in input_addrs_list[:slots_available]:
                    inset.add(addr)
                    input_count += 1
                    input_type_counts[script_type] = input_count_for_type + 1
                    input_distinct_count[script_type] = input_count_distinct_output + 1

                    # Remove from setup sets
                    all_input_addresses.discard(addr)
                    input_addrs.discard(addr)

    return input_count


def third_pass_divide_large_inputs(
    tx,
    all_input_addresses: Set,
    inputs_by_value_and_type: Dict,
    anonsets: Dict,
    avasets: Set,
    show_debug: bool = False
) -> int:
    """
    THIRD PASS: Divide bigger inputs into smaller outputs

    Args:
        tx: CoinjoinTransaction object
        all_input_addresses: Set of unmatched input addresses (modified in-place)
        inputs_by_value_and_type: Dictionary mapping (value, script_type) -> set of addresses
        anonsets: Dictionary of anonymity sets (modified in-place)
        avasets: Set of available output values
        show_debug: Whether to show debug output

    Returns:
        division_count: Number of inputs successfully divided
    """
    # TODO: Implement dynamic programming division algorithm
    # This will be implemented in the next step
    division_count = 0

    if show_debug:
        print("=" * 120)
        print("THIRD PASS: Large Input Division (Not yet implemented)")
        print("=" * 120)
        print(f"Remaining unmatched inputs: {len(all_input_addresses)}")
        print("=" * 120)

    return division_count


def fourth_pass_subset_sum_matching(
    tx,
    all_input_addresses: Set,
    avasets: Set,
    anonsets: Dict,
    show_details: bool = False,
    show_debug: bool = False
) -> Tuple[Dict, Set, int]:
    """
    FOURTH PASS: Subset sum matching - find combinations of inputs that match output values

    Args:
        tx: CoinjoinTransaction object
        all_input_addresses: Set of unmatched input addresses
        avasets: Set of available output values
        anonsets: Dictionary of anonymity sets (modified in-place)
        show_details: Whether to show detailed output
        show_debug: Whether to show debug output

    Returns:
        Tuple of (subset_matches, used_in_combinations, input_count)
    """
    if show_details:
        print("=" * 120)
        print("FOURTH PASS: Subset Sum Matching")
        print("=" * 120)
        print(f"Remaining unmatched inputs: {len(all_input_addresses)}")

    # Get target output values to search for
    target_values = sorted(avasets, reverse=True)
    max_target_value = max(target_values) if target_values else 0

    # Filter inputs - only keep those still unmatched and smaller than max output value
    candidate_inputs = [
        inp for inp in tx.inputs
        if inp.script_pubkey_address in all_input_addresses
        and inp.input_value_satoshi < max_target_value
    ]

    if show_details:
        print(f"Candidate inputs (< max output value): {len(candidate_inputs)}")

    # Configuration
    MAX_COMBINATION_SIZE = 5
    MIN_COMBINATION_SIZE = 2

    # Track subset sum matches
    subset_matches = {}
    used_in_combinations = set()
    input_count = 0

    # Group candidate inputs by script type for same-type combinations
    inputs_by_script_type = {}
    for inp in candidate_inputs:
        if inp.script_pubkey_type not in inputs_by_script_type:
            inputs_by_script_type[inp.script_pubkey_type] = []
        inputs_by_script_type[inp.script_pubkey_type].append(inp)

    if show_details:
        print(f"Input groups by script type: {[(st, len(inps)) for st, inps in inputs_by_script_type.items()]}")
        print()

    # For each target output value
    for target_value in target_values:
        matches_for_this_value = []

        # Try each script type separately (only combine inputs of same type)
        for script_type, inputs_of_type in inputs_by_script_type.items():
            # Filter out inputs already used in other combinations
            available_inputs = [
                inp for inp in inputs_of_type
                if inp.script_pubkey_address not in used_in_combinations
            ]

            # Try different combination sizes
            for combo_size in range(MIN_COMBINATION_SIZE, MAX_COMBINATION_SIZE + 1):
                # Skip if not enough inputs available
                if len(available_inputs) < combo_size:
                    continue

                # Generate combinations of this size
                for combo in combinations(available_inputs, combo_size):
                    # Calculate sum
                    total_value = sum(inp.input_value_satoshi for inp in combo)

                    # Check if it matches the target
                    if total_value == target_value:
                        # Store the match
                        addresses = [inp.script_pubkey_address for inp in combo]
                        script_types = [inp.script_pubkey_type for inp in combo]
                        values = [inp.input_value_satoshi for inp in combo]

                        # Mark these inputs as used (greedy approach)
                        for inp in combo:
                            used_in_combinations.add(inp.script_pubkey_address)

                        matches_for_this_value.append({
                            'addresses': addresses,
                            'script_types': script_types,
                            'values': values,
                            'size': combo_size
                        })

                        # Break after finding first match for this type/size to avoid reusing inputs
                        break

        if matches_for_this_value:
            subset_matches[target_value] = matches_for_this_value
            if show_details:
                print(f"  Value {target_value:,}: Found {len(matches_for_this_value)} matching combinations")

            # Try to add all matches to the anonymity set if there's space
            if show_debug:
                print(f"    [DEBUG] Checking if can add matches to anonymity set...")
                print(f"    [DEBUG] Target value in anonsets: {str(target_value) in anonsets}")

            if str(target_value) in anonsets:
                cur = anonsets[str(target_value)]
                outset = cur["outset"]
                outset: set
                inset = cur["inset"]
                inset: set
                output_type_counts = cur["output_type_counts"]
                input_type_counts = cur["input_type_counts"]
                input_distinct_count = cur["inputs_added_for_distinct_outputs"]

                matches_added = 0
                total_inputs_added = 0

                # Try to add all matches
                for match_idx, match in enumerate(matches_for_this_value):
                    match_addresses = match['addresses']
                    match_script_type = match['script_types'][0]  # All same type in combination

                    if show_debug:
                        print(f"    [DEBUG] Match {match_idx + 1}: {len(match_addresses)} addresses, type: {match_script_type}")

                    # Check if there's space in the anonymity set
                    remaining_slots = len(outset) - len(inset)
                    if show_debug:
                        print(f"    [DEBUG]   Remaining slots: {remaining_slots} (outset: {len(outset)}, inset: {len(inset)})")

                    # Check if output has this script type
                    output_count_for_type = output_type_counts.get(match_script_type, 0)
                    input_count_for_type = input_type_counts.get(match_script_type, 0)
                    input_count_distinct_output = input_distinct_count.get(match_script_type, 0)
                    type_slots_available = output_count_for_type - input_count_distinct_output

                    if show_debug:
                        print(f"    [DEBUG]   Type slots available for {match_script_type}: {type_slots_available} (output: {output_count_for_type}, distinct inputs: {input_count_distinct_output})")

                    # Calculate how many we can add
                    can_add = min(remaining_slots, type_slots_available, len(match_addresses))
                    if show_debug:
                        print(f"    [DEBUG]   Can add: {can_add} (need: {len(match_addresses)})")

                    if can_add > 0:
                        # Add all addresses from this match to the anonymity set
                        for addr in match_addresses:
                            inset.add(addr)
                            input_count += 1
                            total_inputs_added += 1

                        # Update input type count
                        input_type_counts[match_script_type] = input_type_counts.get(match_script_type, 0) + len(match_addresses)
                        input_distinct_count[match_script_type] = input_count_distinct_output + 1

                        matches_added += 1
                        if show_debug:
                            print(f"    ✓ Added match {match_idx + 1}: {len(match_addresses)} inputs to anonymity set")
                    else:
                        if show_debug:
                            print(f"    ✗ Could not add match {match_idx + 1}: insufficient space")
                            if can_add < len(match_addresses):
                                print(f"      Reason: Need {len(match_addresses)} slots but only {can_add} available")

                if show_debug and matches_added > 0:
                    print(f"    Summary: Added {matches_added} match(es) with {total_inputs_added} total inputs to anonymity set")
                elif show_debug:
                    print(f"    Summary: No matches could be added to anonymity set")

            else:
                if show_debug:
                    print(f"    [DEBUG] No anonymity set found for value {target_value}")

    if show_details:
        print(f"\nTotal output values with subset matches: {len(subset_matches)}")
        print("=" * 120)
        print("\n")

    return subset_matches, used_in_combinations, input_count


def print_transaction_side_by_side(tx):
    """
    Print transaction inputs and outputs side by side like mempool.space

    Args:
        tx: CoinjoinTransaction object
    """
    # Get max number of rows needed
    max_rows = max(len(tx.inputs), len(tx.outputs))

    # Column widths
    addr_width = 45
    value_width = 15

    # Header
    print("=" * 130)
    print(f"{'INPUTS':<{addr_width + value_width + 5}} {'OUTPUTS':<{addr_width + value_width}}")
    print(f"{'Address':<{addr_width}} {'Value (sats)':>{value_width}}     {'Address':<{addr_width}} {'Value (sats)':>{value_width}}")
    print("-" * 130)

    # Print rows
    for i in range(max_rows):
        # Input side
        if i < len(tx.inputs):
            inp = tx.inputs[i]
            inp_addr = inp.script_pubkey_address[:42] + "..." if len(inp.script_pubkey_address) > 45 else inp.script_pubkey_address
            inp_value = f"{inp.input_value_satoshi:,}"
        else:
            inp_addr = ""
            inp_value = ""

        # Output side
        if i < len(tx.outputs):
            out = tx.outputs[i]
            out_addr = out.script_pubkey_address[:42] + "..." if len(out.script_pubkey_address) > 45 else out.script_pubkey_address
            out_value = f"{out.output_value_satoshi:,}"
        else:
            out_addr = ""
            out_value = ""

        # Print the row
        print(f"{inp_addr:<{addr_width}} {inp_value:>{value_width}}  →  {out_addr:<{addr_width}} {out_value:>{value_width}}")

    # Footer with totals
    print("-" * 130)
    total_in = sum(inp.input_value_satoshi for inp in tx.inputs)
    total_out = sum(out.output_value_satoshi for out in tx.outputs)
    fee = tx.transaction_fee

    print(f"{'TOTAL:':<{addr_width}} {total_in:>{value_width},}     {'TOTAL:':<{addr_width}} {total_out:>{value_width},}")
    print(f"{'':>{addr_width + value_width + 5}} {'FEE:':<{addr_width}} {fee:>{value_width},}")
    print("=" * 130)


def main():
    """
    Main entry point for the data collection script.
    """

    # ========================================================================
    # COMMAND LINE ARGUMENTS: Parse configuration flags
    # ========================================================================
    parser = argparse.ArgumentParser(description='CoinJoin Anonymity Set Analysis')
    parser.add_argument('-setup', action='store_true', help='Show setup phase output')
    parser.add_argument('-debug', action='store_true', help='Show subset sum debug output')
    parser.add_argument('-v', action='store_true', help='Show subset sum match details')
    parser.add_argument('-sets', action='store_true', help='Show anonymity sets analysis')
    parser.add_argument('-unmatched', action='store_true', help='Show unmatched inputs/outputs')
    parser.add_argument('-tx', action='store_true', help='Show transaction side-by-side view')
    parser.add_argument('--tx-id', type=str, default="EEED55F383A51A566829F58EC229C81C793186923FEDE2C4FE8E0062743A48CB",
                        help='Transaction ID to analyze')

    args = parser.parse_args()

    # ========================================================================
    # CONFIGURATION: Toggle debug output sections
    # ========================================================================
    SHOW_SETUP_PHASE = args.setup
    SHOW_SUBSET_SUM_DEBUG = args.debug
    SHOW_SUBSET_SUM_DETAILS = args.v
    SHOW_ANONYMITY_SETS = args.sets
    SHOW_UNMATCHED = args.unmatched
    SHOW_TRANSACTION_SIDE_BY_SIDE = args.tx

    db_conn = DatabaseConnection()

    tx = db_conn.get_coinjoin_transaction(args.tx_id, True)

    # Print transaction in mempool.space style
    if SHOW_TRANSACTION_SIDE_BY_SIDE:
        print_transaction_side_by_side(tx)

    print("\n")

    # ========================================================================
    # SETUP PHASE: Create sets for easy membership checking and removal
    # ========================================================================
    all_input_addresses, all_output_addresses, inputs_by_value_and_type, outputs_by_value_and_type = setup_phase(tx)

    if SHOW_SETUP_PHASE:
        print("=" * 120)
        print("SETUP PHASE COMPLETE")
        print("=" * 120)
        print(f"Total inputs: {len(tx.inputs)}")
        print(f"Total outputs: {len(tx.outputs)}")
        print(f"Unique input addresses: {len(all_input_addresses)}")
        print(f"Unique output addresses: {len(all_output_addresses)}")
        print(f"Input groups (value, type): {len(inputs_by_value_and_type)}")
        print(f"Output groups (value, type): {len(outputs_by_value_and_type)}")
        print("=" * 120)
        print("\n")

    # ========================================================================
    # FIRST PASS: Build anonymity sets from outputs
    # ========================================================================
    anonsets, avasets, output_count = first_pass_build_anonymity_sets(tx)

    # ========================================================================
    # SECOND PASS: Add inputs to anonymity sets
    # ========================================================================
    input_count = second_pass_add_inputs_to_anonsets(
        inputs_by_value_and_type=inputs_by_value_and_type,
        avasets=avasets,
        anonsets=anonsets,
        all_input_addresses=all_input_addresses
    )

    # ========================================================================
    # THIRD PASS: Divide bigger inputs into smaller outputs
    # ========================================================================
    division_count = third_pass_divide_large_inputs(
        tx=tx,
        all_input_addresses=all_input_addresses,
        inputs_by_value_and_type=inputs_by_value_and_type,
        anonsets=anonsets,
        avasets=avasets,
        show_debug=SHOW_SUBSET_SUM_DETAILS
    )

    # ========================================================================
    # FOURTH PASS: Subset sum matching
    # ========================================================================
    subset_matches, used_in_combinations, subset_input_count = fourth_pass_subset_sum_matching(
        tx=tx,
        all_input_addresses=all_input_addresses,
        avasets=avasets,
        anonsets=anonsets,
        show_details=SHOW_SUBSET_SUM_DETAILS,
        show_debug=SHOW_SUBSET_SUM_DEBUG
    )

    # Add subset sum matched inputs to total count
    input_count += subset_input_count

    # Display subset sum match details
    if SHOW_SUBSET_SUM_DETAILS and subset_matches:
        print("=" * 120)
        print("SUBSET SUM MATCH DETAILS")
        print("=" * 120)

        for target_value in sorted(subset_matches.keys(), reverse=True):
            matches = subset_matches[target_value]
            print(f"\nOutput Value: {target_value:,} satoshis")
            print(f"  Number of matching combinations: {len(matches)}")

            # Show first few combinations (to avoid overwhelming output)
            max_display = 5
            for i, match in enumerate(matches[:max_display]):
                print(f"\n  Combination {i+1} (size: {match['size']}):")
                print(f"    Input values: {[f'{v:,}' for v in match['values']]} = {sum(match['values']):,}")
                print(f"    Script types: {match['script_types']}")
                print(f"    Addresses:")
                for j, addr in enumerate(match['addresses']):
                    print(f"      • {addr} ({match['values'][j]:,} sats)")

            if len(matches) > max_display:
                print(f"\n  ... and {len(matches) - max_display} more combinations")

            print("-" * 120)

        print("=" * 120)
        print("\n")

    # Print anonymity sets in a readable format
    if SHOW_ANONYMITY_SETS:
        print("=" * 120)
        print("ANONYMITY SETS ANALYSIS")
        print("=" * 120)

        for value, anonset_data in sorted(anonsets.items(), key=lambda x: int(x[0]), reverse=True):
            print(f"\nValue: {int(value):,} satoshis")
            print(f"  Script Type: {anonset_data['set_state']}")
            print(f"  Anonymity Set Size: {anonset_data['anon_size']}")
            print(f"  Number of Output Addresses: {len(anonset_data['outset'])}")
            print(f"  Number of Input Addresses: {len(anonset_data['inset'])}")

            # Show script type breakdown with visual indicators
            print(f"\n  Output Script Type Distribution:")
            for script_type, count in sorted(anonset_data['output_type_counts'].items()):
                # Visual indicator: 'p' for taproot, 'q' for segwit
                indicator = 'p' if script_type == 'v1_p2tr' else ('q' if 'v0_p2w' in script_type else '?')
                visual = indicator * count
                print(f"    {script_type}: {count} ({visual})")

            if anonset_data['input_type_counts']:
                print(f"\n  Input Script Type Distribution:")
                for script_type, count in sorted(anonset_data['input_type_counts'].items()):
                    # Visual indicator: 'p' for taproot, 'q' for segwit
                    indicator = 'p' if script_type == 'v1_p2tr' else ('q' if 'v0_p2w' in script_type else '?')
                    visual = indicator * count
                    print(f"    {script_type}: {count} ({visual})")

            if anonset_data['outset']:
                print(f"\n  Output Addresses:")
                for addr in sorted(anonset_data['outset']):
                    # Find the output to get its value
                    matching_out = next((out for out in tx.outputs if out.script_pubkey_address == addr), None)
                    if matching_out:
                        print(f"    → {addr} ({matching_out.output_value_satoshi:,} sats)")
                    else:
                        print(f"    → {addr}")

            if anonset_data['inset']:
                print(f"\n  Input Addresses:")
                for addr in sorted(anonset_data['inset']):
                    # Find the input to get its value
                    matching_inp = next((inp for inp in tx.inputs if inp.script_pubkey_address == addr), None)
                    if matching_inp:
                        print(f"    ← {addr} ({matching_inp.input_value_satoshi:,} sats)")
                    else:
                        print(f"    ← {addr}")

            print("-" * 120)

        print(f"\nTotal unique value amounts: {len(anonsets)}")
        print(f"Total outputs processed: {output_count}")
        print(f"Total inputs matched: {input_count}")
        print("=" * 120)
        print("\n")

    # ========================================================================
    # UNMATCHED INPUTS AND OUTPUTS
    # ========================================================================
    if SHOW_UNMATCHED:
        print("=" * 120)
        print("UNMATCHED INPUTS AND OUTPUTS")
        print("=" * 120)

        # Collect all matched input addresses (from anonsets and subset sum)
        all_matched_inputs = set()
        for anonset_data in anonsets.values():
            all_matched_inputs.update(anonset_data['inset'])
        #all_matched_inputs.update(used_in_combinations)

        # Collect all matched output addresses (from anonsets)
        all_matched_outputs = set()
        for anonset_data in anonsets.values():
            all_matched_outputs.update(anonset_data['outset'])

        # Find unmatched inputs and outputs
        unmatched_inputs = [inp for inp in tx.inputs if inp.script_pubkey_address not in all_matched_inputs]
        unmatched_outputs = [out for out in tx.outputs if out.script_pubkey_address not in all_matched_outputs]

        print(f"\nUnmatched Inputs: {len(unmatched_inputs)}")
        print(f"Unmatched Outputs: {len(unmatched_outputs)}")
        print()

        if unmatched_inputs:
            print("UNMATCHED INPUT ADDRESSES:")
            print("-" * 120)
            for inp in sorted(unmatched_inputs, key=lambda x: x.input_value_satoshi, reverse=True):
                script_indicator = 'p' if inp.script_pubkey_type == 'v1_p2tr' else ('q' if 'v0_p2w' in inp.script_pubkey_type else '?')
                print(f"  ← [{script_indicator}] {inp.script_pubkey_address} ({inp.input_value_satoshi:,} sats) - {inp.script_pubkey_type}")
            print()

        if unmatched_outputs:
            print("UNMATCHED OUTPUT ADDRESSES:")
            print("-" * 120)
            for out in sorted(unmatched_outputs, key=lambda x: x.output_value_satoshi, reverse=True):
                script_indicator = 'p' if out.script_pubkey_type == 'v1_p2tr' else ('q' if 'v0_p2w' in out.script_pubkey_type else '?')
                print(f"  → [{script_indicator}] {out.script_pubkey_address} ({out.output_value_satoshi:,} sats) - {out.script_pubkey_type}")
            print()

        # Summary statistics
        total_unmatched_input_value = sum(inp.input_value_satoshi for inp in unmatched_inputs)
        total_unmatched_output_value = sum(out.output_value_satoshi for out in unmatched_outputs)

        print("SUMMARY:")
        print("-" * 120)
        print(f"Total unmatched input value:  {total_unmatched_input_value:,} sats")
        print(f"Total unmatched output value: {total_unmatched_output_value:,} sats")
        print(f"Difference: {total_unmatched_input_value - total_unmatched_output_value:,} sats")
        print("=" * 120)


    

        



    
    


if __name__ == "__main__":
    main()