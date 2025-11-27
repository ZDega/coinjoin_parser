"""
CoinJoin Transaction Data Collection Script

This script collects CoinJoin transaction data from Bitcoin APIs and stores it
in the DuckDB database according to the schema defined in data/create_data_base.py
"""

import os
from typing import List, Dict, Any, Optional, Tuple
from dotenv import load_dotenv
import duckdb
from tqdm import tqdm

# Import existing API client
from api_clients.mempool_client import MempoolClient

load_dotenv()


# ============================================================================
# DATABASE CONNECTION
# ============================================================================

class DatabaseConnection:
    """Manages DuckDB database connection and operations."""

    def __init__(self, db_path: Optional[str] = None):
        """
        Initialize database connection.

        Args:
            db_path: Path to DuckDB database. If None, uses CLUSTERING_DATABASE_PATH env var.
        """
        self.db_path = db_path or os.getenv("CLUSTERING_DATABASE_PATH")
        self.conn = duckdb.connect(self.db_path)

    def __enter__(self) -> "DatabaseConnection":
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        """Context manager exit - close connection."""
        if self.conn:
            self.conn.close()
        return False  # Don't suppress exceptions - let them propagate



    def get_or_create_coordinator_id(self, coordinator_endpoint: str) -> int:
        """
        Get coordinator ID or create new coordinator entry.

        Args:
            coordinator_endpoint: URL of the coordinator

        Returns:
            Coordinator ID
        """
        # Check if coordinator already exists
        result = self.conn.execute(
            "SELECT coor_id FROM coordinators WHERE coordinator_endpoint = ?",
            [coordinator_endpoint]
        ).fetchone()

        if result:
            # Coordinator exists, return its ID
            return result[0]

        # Coordinator doesn't exist, create new entry
        self.conn.execute(
            "INSERT INTO coordinators (coordinator_endpoint) VALUES (?)",
            [coordinator_endpoint]
        )

        # Get the newly created coordinator ID
        result = self.conn.execute(
            "SELECT coor_id FROM coordinators WHERE coordinator_endpoint = ?",
            [coordinator_endpoint]
        ).fetchone()

        return result[0]
    def get_input_address(self, address: str, script_type: str) -> Optional[Tuple[int, str, bool, str, int, int]]: #TODO implement pydantic objects for these
        """
        Get input address ID or if doesn't exist returns Null.

        Args:
            address: scriptpubkey_address in str form
            script_type: Script pubkey type (p2pkh, p2sh, v0_p2wpkh, etc.)

        Returns:
            Tuple of (input_address_id, address_hex, used_as_output, script_type, number_of_cjs_used_in_as_input, total_amount_spent_in_cj)
            or None if address doesn't exist

        Raises:
            ValueError: If the stored script_type doesn't match the provided script_type
        """
        # Check if input address already exists
        result = self.conn.execute(
            "SELECT input_address_id, LOWER(to_hex(address)), used_as_output, script_type, number_of_cjs_used_in_as_input, total_amount_spent_in_cj FROM input_addresses WHERE address = from_hex(?)",
            [address]
        ).fetchone()
        if result:
            # Verify script_type matches
            stored_script_type = result[3]
            if stored_script_type != script_type:
                raise ValueError(
                    f"Script type mismatch for address {result[1]}: "
                    f"stored='{stored_script_type}', provided='{script_type}'"
                )
            return result

        return None
    
    def get_output_address(self, address: str, script_type: str) -> Optional[Tuple[int, str, bool, str, int, int]]: #TODO implement pydantic objects for these
        """
        Get input address ID or if doesn't exist returns Null.

        Args:
            address: scriptpubkey_address in str form
            script_type: Script pubkey type (p2pkh, p2sh, v0_p2wpkh, etc.)

        Returns:
            Tuple of (input_address_id, address_hex, used_as_output, script_type, number_of_cjs_used_in_as_input, total_amount_spent_in_cj)
            or None if address doesn't exist

        Raises:
            ValueError: If the stored script_type doesn't match the provided script_type
        """
        # Check if input address already exists
        result = self.conn.execute(
            "SELECT output_address_id, LOWER(to_hex(address)), used_as_input, script_type, number_of_cjs_used_in_as_output, total_amount_received_in_cj FROM output_addresses WHERE address = from_hex(?)",
            [address]
        ).fetchone()
        if result:
            # Verify script_type matches
            stored_script_type = result[3]
            if stored_script_type != script_type:
                raise ValueError(
                    f"Script type mismatch for address {result[1]}: "
                    f"stored='{stored_script_type}', provided='{script_type}'"
                )
            return result

        return None

    def update_or_insert_input_address(self, address: str, script_type: str, value: int) -> Tuple[int, str, bool, str, int, int]:
        """
        Get input address ID or create new input address entry.

        Args:
            address: scriptpubkey_address in str form
            script_type: Script pubkey type (p2pkh, p2sh, v0_p2wpkh, etc.)
            value: Input value in satoshis to add to total_amount_spent_in_cj

        Returns:
            Tuple of (input_address_id, address_hex, used_as_output, script_type, number_of_cjs_used_in_as_input, total_amount_spent_in_cj)
        """
        # Check if input address already exists
        result = self.get_input_address(address, script_type)

        if result:
            # Address exists, update statistics
            self.conn.execute(
                """UPDATE input_addresses
                SET number_of_cjs_used_in_as_input = number_of_cjs_used_in_as_input + 1,
                total_amount_spent_in_cj = total_amount_spent_in_cj + ?
                WHERE input_address_id = ?""",
                [value, result[0]]
            )
            # Return updated tuple (tuples are immutable, so create a new one with updated values)
            return (result[0], result[1], result[2], result[3], result[4] + 1, result[5] + value)

        # Address doesn't exist, create new entry
        self.conn.execute(
            """INSERT INTO input_addresses (address, used_as_output, script_type, number_of_cjs_used_in_as_input, total_amount_spent_in_cj)
               VALUES (?, FALSE, ?::script_pubkey_type, 1, ?)""",
            [address, script_type, value]
        )

        # Get the newly created input address with full data
        result = self.conn.execute(
            "SELECT input_address_id, LOWER(to_hex(address)), used_as_output, script_type, number_of_cjs_used_in_as_input, total_amount_spent_in_cj FROM input_addresses WHERE address = from_hex(?)",
            [address]
        ).fetchone()

        return result
    

    def update_or_insert_output_address(self, address: str, script_type: str, value: int) -> Tuple[int, str, bool, str, int, int]:
        """
        Get input address ID or create new input address entry.

        Args:
            address: scriptpubkey_address in str
            script_type: Script pubkey type (p2pkh, p2sh, v0_p2wpkh, etc.)
            value: Input value in satoshis to add to total_amount_spent_in_cj

        Returns:
            Tuple of (input_address_id, address_hex, used_as_output, script_type, number_of_cjs_used_in_as_input, total_amount_spent_in_cj)
        """
        # Check if input address already exists
        result = self.get_output_address(address, script_type)

        if result:
            # Address exists, update statistics
            self.conn.execute(
                """UPDATE output_addresses
                SET number_of_cjs_used_in_as_output = number_of_cjs_used_in_as_output + 1,
                total_amount_received_in_cj = total_amount_received_in_cj + ?
                WHERE output_address_id = ?""",
                [value, result[0]]
            )
            # Return updated tuple (tuples are immutable, so create a new one with updated values)
            return (result[0], result[1], result[2], result[3], result[4] + 1, result[5] + value)

        # Address doesn't exist, create new entry
        self.conn.execute(
            """INSERT INTO output_addresses (address, used_as_input, script_type, number_of_cjs_used_in_as_output, total_amount_received_in_cj)
               VALUES (?, FALSE, ?::script_pubkey_type, 1, ?)""",
            [address, script_type, value]
        )

        # Get the newly created input address with full data
        result = self.conn.execute(
            "SELECT output_address_id, LOWER(to_hex(address)), used_as_input, script_type, number_of_cjs_used_in_as_output, total_amount_received_in_cj FROM output_addresses WHERE address = from_hex(?)",
            [address]
        ).fetchone()

        return result

    def get_coinjoin_transaction(self, tx_id) -> Dict[str,Any]:
        pass
    def insert_coinjoin_transaction(self, tx_data: Dict[str, Any]) -> bool: #TODO create a pydantic object for this
        """
        Insert CoinJoin transaction into database.

        Args:
            tx_data: Processed transaction data matching coinjoin_transactions schema

        Returns:
            True if successful, False otherwise
        """
        # TODO: Implement transaction insertion
        # Steps:
        # 1. Prepare INSERT statement for coinjoin_transactions table
        # 2. Handle all fields from the schema:
        #    - tx_id (BLOB)
        #    - number_inputs, number_outputs
        #    - value_inputs, value_outputs
        #    - inputs (transaction_input[])
        #    - outputs (transaction_output[])
        #    - coordinator_id
        #    - transaction_fee, block_number, block_time
        #    - raw_size_bytes, weight, fee_rate_sat_per_vbyte
        # 3. Execute INSERT (handle UNIQUE constraint on tx_id)
        # 4. Return success/failure status
        pass


# ============================================================================
# DATA PROCESSING
# ============================================================================

class TransactionProcessor:
    """Processes raw transaction data into database schema format."""

    @staticmethod
    def detect_script_type(scriptpubkey_type: str) -> str:
        """
        Map API script type to database enum.

        Args:
            scriptpubkey_type: Script type from API (e.g., 'v1_p2tr', 'v0_p2wpkh')

        Returns:
            Script type enum value matching script_pubkey_type enum
        """
        # TODO: Implement script type mapping
        # Map API script types to database enum values:
        # - 'p2pkh' -> 'p2pkh'
        # - 'p2sh' -> 'p2sh'
        # - 'v0_p2wpkh' -> 'v0_p2wpkh'
        # - 'v0_p2wsh' -> 'v0_p2wsh'
        # - 'v1_p2tr' -> 'v1_p2tr'
        # - anything else -> 'other'
        pass

    @staticmethod
    def txid_to_bytes(txid: str) -> bytes:
        """
        Convert transaction ID hex string to bytes.

        Args:
            txid: Transaction ID as hex string

        Returns:
            Transaction ID as bytes
        """
        # TODO: Implement txid conversion
        # - Convert hex string to bytes
        # - Handle potential errors
        pass

    @staticmethod
    def address_to_bytes(address: str) -> bytes:
        """
        Convert scriptpubkey address to bytes for storage.

        Args:
            address: Bitcoin address or scriptpubkey hex

        Returns:
            Address in binary form
        """
        # TODO: Implement address conversion
        # - If address is hex string, convert to bytes
        # - Store raw scriptpubkey as bytes
        # - Handle empty/None addresses
        pass

    @staticmethod
    def process_input(vin: Dict[str, Any], input_address_id: int) -> Dict[str, Any]:
        """
        Process transaction input into transaction_input struct format.

        Args:
            vin: Raw input data from mempool API
            input_address_id: ID from input_addresses table

        Returns:
            Processed input matching transaction_input struct schema
        """
        # TODO: Implement input processing
        # Required fields for transaction_input struct:
        # - prev_tx_id: BLOB (from vin['txid'])
        # - prev_vout_index: INTEGER (from vin['vout'])
        # - script_pubkey_type: script_pubkey_type enum (from vin['prevout']['scriptpubkey_type'])
        # - script_pubkey_address: BLOB (from vin['prevout']['scriptpubkey'])
        # - input_value_satoshi: BIGINT (from vin['prevout']['value'])
        # - is_coinbase: BOOL (from vin['is_coinbase'])
        # - input_address_id: INTEGER (passed as parameter)
        pass

    @staticmethod
    def process_output(vout: Dict[str, Any], output_address_id: int) -> Dict[str, Any]:
        """
        Process transaction output into transaction_output struct format.

        Args:
            vout: Raw output data from mempool API
            output_address_id: ID from output_addresses table

        Returns:
            Processed output matching transaction_output struct schema
        """
        # TODO: Implement output processing
        # Required fields for transaction_output struct:
        # - vout_index: INTEGER (index in vout array)
        # - script_pubkey_type: script_pubkey_type enum (from vout['scriptpubkey_type'])
        # - script_pubkey_address: BLOB (from vout['scriptpubkey'])
        # - output_value_satoshi: BIGINT (from vout['value'])
        # - output_address_id: INTEGER (passed as parameter)
        pass

    @staticmethod
    def process_transaction(tx_raw: Dict[str, Any], coordinator_id: int,
                           db_conn: DatabaseConnection) -> Dict[str, Any]:
        """
        Process raw transaction data into database schema format.

        Args:
            tx_raw: Raw transaction data from mempool API
            coordinator_id: Coordinator ID from database
            db_conn: Database connection for address lookups

        Returns:
            Processed transaction data ready for insertion
        """
        # TODO: Implement full transaction processing
        # Steps:
        # 1. Extract basic transaction info:
        #    - tx_id: bytes (from tx_raw['txid'])
        #    - number_inputs: len(tx_raw['vin'])
        #    - number_outputs: len(tx_raw['vout'])
        #
        # 2. Process inputs:
        #    - For each input in tx_raw['vin']:
        #      a. Extract address and script type
        #      b. Get/create input_address_id from database
        #      c. Process input with process_input()
        #    - Calculate value_inputs (sum of all input values)
        #
        # 3. Process outputs:
        #    - For each output in tx_raw['vout']:
        #      a. Extract address and script type
        #      b. Get/create output_address_id from database
        #      c. Process output with process_output()
        #    - Calculate value_outputs (sum of all output values)
        #
        # 4. Calculate derived fields:
        #    - transaction_fee: value_inputs - value_outputs
        #    - fee_rate_sat_per_vbyte: fee / (weight / 4)
        #
        # 5. Extract block info:
        #    - block_number: tx_raw['status']['block_height']
        #    - block_time: tx_raw['status']['block_time'] (convert to timestamp)
        #
        # 6. Extract size/weight:
        #    - raw_size_bytes: tx_raw['size']
        #    - weight: tx_raw['weight']
        #
        # 7. Add coordinator_id
        #
        # 8. Return complete dictionary matching coinjoin_transactions schema
        pass

    @staticmethod
    def is_coinjoin_transaction(tx_data: Dict[str, Any]) -> bool:
        """
        Determine if a transaction is a CoinJoin transaction.

        Args:
            tx_data: Raw transaction data from API

        Returns:
            True if transaction appears to be a CoinJoin
        """
        # TODO: Implement CoinJoin detection heuristics
        # Common CoinJoin characteristics:
        # 1. Multiple inputs (typically >= 2)
        # 2. Multiple outputs (typically >= 3)
        # 3. Equal-valued outputs (common pattern in Wasabi, Whirlpool, JoinMarket)
        # 4. Output values follow specific patterns
        #
        # Simple heuristic to start:
        # - Check if there are at least 2 equal-valued outputs
        # - Check if number of inputs >= 2 and outputs >= 3
        pass


# ============================================================================
# MAIN COLLECTION WORKFLOW
# ============================================================================

class DataCollector:
    """Main data collection coordinator."""

    def __init__(self, coordinator_endpoint: str):
        """
        Initialize data collector.

        Args:
            coordinator_endpoint: CoinJoin coordinator endpoint (for tracking purposes)
        """
        # TODO: Initialize collector
        # - Store coordinator_endpoint
        # - Initialize MempoolClient
        # - Initialize DatabaseConnection
        # - Initialize TransactionProcessor
        pass

    def collect_transaction(self, txid: str) -> bool:
        """
        Collect and store a single transaction.

        Args:
            txid: Transaction ID to collect

        Returns:
            True if successful, False otherwise
        """
        # TODO: Implement single transaction collection
        # Steps:
        # 1. Fetch transaction using mempool_client.get_transaction(txid)
        # 2. Check if it's a CoinJoin using TransactionProcessor.is_coinjoin_transaction()
        # 3. If not CoinJoin, return False
        # 4. Get/create coordinator_id from database
        # 5. Process transaction using TransactionProcessor.process_transaction()
        # 6. Insert into database using db_conn.insert_coinjoin_transaction()
        # 7. Return success status
        pass

    def collect_transactions_batch(self, txids: List[str], show_progress: bool = True) -> Dict[str, int]:
        """
        Collect multiple transactions in batch.

        Args:
            txids: List of transaction IDs to collect
            show_progress: Whether to show progress bar (default: True)

        Returns:
            Dictionary with statistics: {'collected': N, 'skipped': M, 'errors': K}
        """
        # TODO: Implement batch collection
        # Steps:
        # 1. Initialize counters for collected, skipped, errors
        # 2. Create progress bar with tqdm:
        #    - If show_progress: use tqdm(txids, desc="Collecting transactions")
        #    - Otherwise: iterate normally over txids
        # 3. For each txid:
        #    - Try to collect_transaction()
        #    - Update appropriate counter
        #    - Update progress bar description with current stats
        #    - Handle errors gracefully
        # 4. Return statistics dictionary
        #
        # Example with progress bar:
        #   stats = {'collected': 0, 'skipped': 0, 'errors': 0}
        #   iterator = tqdm(txids, desc="Collecting") if show_progress else txids
        #   for txid in iterator:
        #       try:
        #           success = self.collect_transaction(txid)
        #           if success:
        #               stats['collected'] += 1
        #           else:
        #               stats['skipped'] += 1
        #           if show_progress:
        #               iterator.set_postfix(collected=stats['collected'],
        #                                    skipped=stats['skipped'],
        #                                    errors=stats['errors'])
        #       except Exception as e:
        #           stats['errors'] += 1
        #   return stats
        pass

    def collect_from_txid_list_file(self, file_path: str, show_progress: bool = True) -> Dict[str, int]:
        """
        Collect transactions from a file containing transaction IDs (one per line).

        Args:
            file_path: Path to file with transaction IDs
            show_progress: Whether to show progress bar (default: True)

        Returns:
            Dictionary with statistics
        """
        # TODO: Implement file-based collection
        # Steps:
        # 1. Read transaction IDs from file (one per line, strip whitespace)
        # 2. Filter out empty lines and comments (lines starting with #)
        # 3. Call collect_transactions_batch() with the list and show_progress
        # 4. Return statistics
        #
        # Example:
        #   with open(file_path, 'r') as f:
        #       txids = [line.strip() for line in f
        #                if line.strip() and not line.strip().startswith('#')]
        #   return self.collect_transactions_batch(txids, show_progress=show_progress)
        pass


# ============================================================================
# CLI INTERFACE
# ============================================================================

def main():
    """Main entry point for the data collection script."""
    # TODO: Implement CLI
    # Arguments needed:
    # - coordinator_endpoint: Name/URL of coordinator for tracking
    # - mode: 'single', 'batch', 'file'
    # - input: txid, list of txids, or file path
    #
    # Example usage:
    # python cj_tx_data_collection.py --coordinator "wasabi" --mode single --txid abc123...
    # python cj_tx_data_collection.py --coordinator "wasabi" --mode file --input txids.txt
    #
    # Steps:
    # 1. Parse command line arguments (use argparse)
    # 2. Initialize DataCollector with coordinator_endpoint
    # 3. Based on mode:
    #    - single: call collect_transaction()
    #    - batch: call collect_transactions_batch()
    #    - file: call collect_from_txid_list_file()
    # 4. Print summary statistics
    # 5. Handle errors and provide meaningful error messages
    pass


if __name__ == "__main__":
    main()
