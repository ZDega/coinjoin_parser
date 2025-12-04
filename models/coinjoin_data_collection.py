"""
CoinJoin Transaction Data Collection Script

This script collects CoinJoin transaction data from Bitcoin APIs and stores it
in the DuckDB database according to the schema defined in data/create_data_base.py
"""

import os
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv
import duckdb
from tqdm import tqdm

# Import existing API client
from api_clients.mempool_client import MempoolClient

# Import Pydantic models
from models.coinjoin_data import (
    ScriptPubkeyType,
    TransactionInput,
    TransactionOutput,
    Coordinator,
    InputAddress,
    OutputAddress,
    CoinjoinTransaction,
)

load_dotenv()


# ============================================================================
# DATABASE CONNECTION
# ============================================================================


class DatabaseConnection:
    """Manages DuckDB database connection and operations."""

    def __init__(self, db_path: Optional[str] = None, db_memory: bool = False, connection: Optional[duckdb.DuckDBPyConnection] = None):
        """
        Initialize database connection.

        Args:
            db_path: Path to DuckDB database. If None, uses CLUSTERING_DATABASE_PATH env var.
            db_memory: If True, creates an in-memory database (overrides db_path).
            connection: Existing DuckDB connection to reuse. If provided, db_path and db_memory are ignored.
        """
        if connection is not None:
            # Use existing connection
            self.conn = connection
            self.db_path = None
            self._owns_connection = False
        elif db_memory:
            # Create in-memory database
            self.conn = duckdb.connect(":memory:")
            self.db_path = ":memory:"
            self._owns_connection = True
        else:
            # Use provided path or fall back to environment variable
            self.db_path = db_path or os.getenv("CLUSTERING_DATABASE_PATH")
            self.conn = duckdb.connect(self.db_path)
            self._owns_connection = True

    def __enter__(self) -> "DatabaseConnection":
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        """Context manager exit - close connection only if we own it."""
        if self.conn and self._owns_connection:
            self.conn.close()
        return False  # Don't suppress exceptions - let them propagate

    def get_or_create_coordinator(self, coordinator_endpoint: str) -> Coordinator:
        """
        Get coordinator or create new coordinator entry if it doesn't exist.

        Args:
            coordinator_endpoint: URL of the coordinator

        Returns:
            Coordinator object with ID and endpoint
        """
        # Check if coordinator already exists
        result = self.conn.execute(
            "SELECT coor_id, coordinator_endpoint FROM coordinators WHERE coordinator_endpoint = ?",
            [coordinator_endpoint]
        ).fetchone()

        if result:
            # Coordinator exists, return as Coordinator object
            print(f"  🔗 Found existing coordinator (ID: {result[0]})")
            return Coordinator.from_db_row(*result)

        # Coordinator doesn't exist, create new entry
        self.conn.execute(
            "INSERT INTO coordinators (coordinator_endpoint) VALUES (?)",
            [coordinator_endpoint]
        )

        # Get the newly created coordinator
        result = self.conn.execute(
            "SELECT coor_id, coordinator_endpoint FROM coordinators WHERE coordinator_endpoint = ?",
            [coordinator_endpoint]
        ).fetchone()

        print(f"  ➕ Created new coordinator (ID: {result[0]})")
        return Coordinator.from_db_row(*result)

    def get_or_create_coordinator_id(self, coordinator_endpoint: str) -> int:
        """
        Get coordinator ID, creating the coordinator if it doesn't exist.

        Args:
            coordinator_endpoint: URL of the coordinator

        Returns:
            Coordinator ID
        """
        return self.get_or_create_coordinator(coordinator_endpoint=coordinator_endpoint).coor_id

    def get_input_address(self, address: str, script_type: str) -> Optional[InputAddress]:
        """
        Get input address from database if it exists.

        Args:
            address: Script pubkey address (Bitcoin address format)
            script_type: Script pubkey type (p2pkh, p2sh, v0_p2wpkh, etc.)

        Returns:
            InputAddress object if exists, None otherwise

        Raises:
            ValueError: If the stored script_type doesn't match the provided script_type
        """
        
        

        # Query for existing input address
        result = self.conn.execute(
            """SELECT input_address_id, LOWER(to_hex(address)), used_as_output, script_type,
                      number_of_cjs_used_in_as_input, total_amount_spent_in_cj
               FROM input_addresses
               WHERE address = ?""",
            [address]
        ).fetchone()

        if result:
            # Verify script_type matches to maintain data integrity
            stored_script_type = result[3]
            if stored_script_type != script_type:
                raise ValueError(
                    f"Script type mismatch for address {result[1]}: "
                    f"stored='{stored_script_type}', provided='{script_type}'"
                )
            return InputAddress.from_db_row(*result)

        return None

    def get_output_address(self, address: str, script_type: str) -> Optional[OutputAddress]:
        """
        Get output address from database if it exists.

        Args:
            address: Script pubkey address (Bitcoin address format)
            script_type: Script pubkey type (p2pkh, p2sh, v0_p2wpkh, etc.)

        Returns:
            OutputAddress object if exists, None otherwise

        Raises:
            ValueError: If the stored script_type doesn't match the provided script_type
        """
        

        # Query for existing output address
        result = self.conn.execute(
            """SELECT output_address_id, LOWER(to_hex(address)), used_as_input, script_type,
                      number_of_cjs_used_in_as_output, total_amount_received_in_cj
               FROM output_addresses
               WHERE address = ?""",
            [address]
        ).fetchone()

        if result:
            # Verify script_type matches to maintain data integrity
            stored_script_type = result[3]
            if stored_script_type != script_type:
                raise ValueError(
                    f"Script type mismatch for address {result[1]}: "
                    f"stored='{stored_script_type}', provided='{script_type}'"
                )
            return OutputAddress.from_db_row(*result)

        return None

    def update_or_insert_input_address(self, address: str, script_type: str, value: int) -> InputAddress:
        """
        Update existing input address statistics or create new entry if it doesn't exist.

        This method increments usage counters and updates total amounts for addresses
        used as inputs in CoinJoin transactions.

        Args:
            address: Script pubkey address (Bitcoin address format)
            script_type: Script pubkey type (p2pkh, p2sh, v0_p2wpkh, etc.)
            value: Input value in satoshis to add to total_amount_spent_in_cj

        Returns:
            InputAddress object with updated values
        """
        # Check if input address already exists
        result = self.get_input_address(address, script_type)

        if result:
            # Address exists, update statistics in database
            self.conn.execute(
                """UPDATE input_addresses
                   SET number_of_cjs_used_in_as_input = number_of_cjs_used_in_as_input + 1,
                       total_amount_spent_in_cj = total_amount_spent_in_cj + ?
                   WHERE input_address_id = ?""",
                [value, result.input_address_id]
            )
            print(f"    📝 Updated input address {address[:16]}... (ID: {result.input_address_id})")
            # Return new object with updated values
            return InputAddress.from_db_row(
                result.input_address_id,
                result.address,
                result.used_as_output,
                result.script_type,
                result.number_of_cjs_used_in_as_input + 1,
                result.total_amount_spent_in_cj + value
            )

        # Address doesn't exist, create new entry
        #print(f"    ➕ Inserting new input address {address[:16]}... into input_addresses table")
        
        
        self.conn.execute(
            """INSERT INTO input_addresses (address, used_as_output, script_type,
                                             number_of_cjs_used_in_as_input, total_amount_spent_in_cj)
               VALUES (?, FALSE, ?::script_pubkey_type, 1, ?)""",
            [address, script_type, value]
        )

        # Get the newly created input address with full data
        result = self.conn.execute(
            """SELECT input_address_id, LOWER(to_hex(address)), used_as_output, script_type,
                      number_of_cjs_used_in_as_input, total_amount_spent_in_cj
               FROM input_addresses
               WHERE address = ?""",
            [address]
        ).fetchone()

        #print(f"    ✅ Created new input address with ID: {result[0]}")
        return InputAddress.from_db_row(*result)

    def update_or_insert_output_address(self, address: str, script_type: str, value: int) -> OutputAddress:
        """
        Update existing output address statistics or create new entry if it doesn't exist.

        This method increments usage counters and updates total amounts for addresses
        used as outputs in CoinJoin transactions.

        Args:
            address: Script pubkey address (Bitcoin address format)
            script_type: Script pubkey type (p2pkh, p2sh, v0_p2wpkh, etc.)
            value: Output value in satoshis to add to total_amount_received_in_cj

        Returns:
            OutputAddress object with updated values
        """
        # Check if output address already exists
        result = self.get_output_address(address, script_type)

        if result:
            # Address exists, update statistics in database
            self.conn.execute(
                """UPDATE output_addresses
                   SET number_of_cjs_used_in_as_output = number_of_cjs_used_in_as_output + 1,
                       total_amount_received_in_cj = total_amount_received_in_cj + ?
                   WHERE output_address_id = ?""",
                [value, result.output_address_id]
            )
            print(f"    📝 Updated output address {address[:16]}... (ID: {result.output_address_id})")
            # Return new object with updated values
            return OutputAddress.from_db_row(
                result.output_address_id,
                result.address,
                result.used_as_input,
                result.script_type,
                result.number_of_cjs_used_in_as_output + 1,
                result.total_amount_received_in_cj + value
            )

        # Address doesn't exist, create new entry
        #print(f"    ➕ Inserting new output address {address[:16]}... into output_addresses table")

        self.conn.execute(
            """INSERT INTO output_addresses (address, used_as_input, script_type,
                                              number_of_cjs_used_in_as_output, total_amount_received_in_cj)
               VALUES (?, FALSE, ?::script_pubkey_type, 1, ?)""",
            [address, script_type, value]
        )

        # Get the newly created output address with full data
        result = self.conn.execute(
            """SELECT output_address_id, LOWER(to_hex(address)), used_as_input, script_type,
                      number_of_cjs_used_in_as_output, total_amount_received_in_cj
               FROM output_addresses
               WHERE address = ?""",
            [address]
        ).fetchone()

        #print(f"    ✅ Created new output address with ID: {result[0]}")
        return OutputAddress.from_db_row(*result)

    def get_coinjoin_transaction(self, tx_id: str, show_progress: Optional[str] = False) -> Optional[CoinjoinTransaction]:
        """
        Retrieve a CoinJoin transaction from the database by transaction ID.

        Args:
            tx_id: Transaction ID in hex string format

        Returns:
            CoinjoinTransaction object if found, None otherwise
        """
        # Query for existing coinjoin transaction
        result = self.conn.execute(
            """SELECT tx_id_int, LOWER(to_hex(tx_id)), number_inputs, number_outputs,
                      value_inputs, value_outputs, inputs, outputs, coordinator_id,
                      transaction_fee, block_number, block_time, raw_size_bytes,
                      weight, fee_rate_sat_per_vbyte, processed
               FROM coinjoin_transactions
               WHERE tx_id = from_hex(?)""",
            [tx_id]
        ).fetchone()

        if result:
            # Convert DuckDB struct arrays to Pydantic models
            inputs = TransactionInput.from_db_rows(result[6], show_progress)
            outputs = TransactionOutput.from_db_rows(result[7], show_progress)

            # Create and return CoinjoinTransaction with all fields
            return CoinjoinTransaction.from_db_row(
                tx_id_int=result[0],
                tx_id=result[1],
                number_inputs=result[2],
                number_outputs=result[3],
                value_inputs=result[4],
                value_outputs=result[5],
                inputs=inputs,
                outputs=outputs,
                coordinator_id=result[8],
                transaction_fee=result[9],
                block_number=result[10],
                block_time=result[11],
                raw_size_bytes=result[12],
                weight=result[13],
                fee_rate_sat_per_vbyte=result[14],
                processed=result[15]
            )

        return None

    def insert_coinjoin_transaction(self, tx_data: CoinjoinTransaction) -> bool:
        """
        Insert CoinJoin transaction into database.

        Args:
            tx_data: CoinjoinTransaction Pydantic model with all transaction data

        Returns:
            True if successful, False otherwise
        """
        try:
            # Convert Pydantic model to DuckDB-compatible format
            db_data = CoinjoinTransaction.from_object_to_db(tx_data)

            # Insert transaction into database
            self.conn.execute(
                """INSERT INTO coinjoin_transactions (
                    tx_id, number_inputs, number_outputs, value_inputs, value_outputs,
                    inputs, outputs, coordinator_id, transaction_fee, block_number,
                    block_time, raw_size_bytes, weight, fee_rate_sat_per_vbyte, processed
                ) VALUES (
                    ?, ?, ?, ?, ?,
                    ?::transaction_input[], ?::transaction_output[], ?, ?, ?,
                    ?, ?, ?, ?, ?
                )""",
                [
                    db_data["tx_id"],
                    db_data["number_inputs"],
                    db_data["number_outputs"],
                    db_data["value_inputs"],
                    db_data["value_outputs"],
                    db_data["inputs"],
                    db_data["outputs"],
                    db_data["coordinator_id"],
                    db_data["transaction_fee"],
                    db_data["block_number"],
                    db_data["block_time"],
                    db_data["raw_size_bytes"],
                    db_data["weight"],
                    db_data["fee_rate_sat_per_vbyte"],
                    db_data["processed"]
                ]
            )
            
            return True

        except Exception as e: #TODO comprehensive error handling
            # Handle duplicate key or other database errors
            print(f"  ❌ Error inserting transaction {tx_data.tx_id}: {e}")
            return False


# ============================================================================
# DATA PROCESSING
# ============================================================================


class TransactionProcessor:
    """Processes raw transaction data into database schema format."""

    @staticmethod
    def detect_script_type(scriptpubkey_type: str) -> str:
        """
        Map API script type to database enum value.

        Args:
            scriptpubkey_type: Script type from API (e.g., 'v1_p2tr', 'v0_p2wpkh')

        Returns:
            Script type enum value matching script_pubkey_type enum
        """
        # Normalize to lowercase for case-insensitive matching
        script_type_lower = scriptpubkey_type.lower().strip()

        # Map of known script types
        valid_types = {
            'p2pkh',
            'p2sh',
            'v0_p2wpkh',
            'v0_p2wsh',
            'v1_p2tr'
        }

        # Return the script type if it's valid, otherwise return 'other'
        if script_type_lower in valid_types:
            return script_type_lower
        else:
            print(f"⚠️  Warning: Unknown script type '{scriptpubkey_type}' classified as 'other' - consider adding support for this type")
            return 'other'

    @staticmethod
    def txid_to_hex(txid: str) -> str:
        """
        Normalize transaction ID to hex string format.

        Args:
            txid: Transaction ID as hex string

        Returns:
            Normalized transaction ID as hex string (lowercase, 64 characters)

        Raises:
            ValueError: If txid is not a valid 64-character hex string
        """
        # Strip whitespace and convert to lowercase
        normalized_txid = txid.strip().lower()

        # Validate length (Bitcoin txid is 32 bytes = 64 hex characters)
        if len(normalized_txid) != 64:
            raise ValueError(
                f"Invalid transaction ID length: expected 64 characters, got {len(normalized_txid)} "
                f"(txid: '{txid}')"
            )

        # Validate hex format
        try:
            int(normalized_txid, 16)
        except ValueError:
            raise ValueError(
                f"Invalid transaction ID format: must be hexadecimal string (txid: '{txid}')"
            )

        return normalized_txid

    @staticmethod
    def process_input(vin: Dict[str, Any], input_address_id: int) -> TransactionInput: 
        #TODO change the function according to the style of the DataCollector
        #TODO as the input_address_id is generated after an input is proccesed 
        #TODO may be pass the DatabaseConnection class into the funciton 
        """
        Process transaction input into TransactionInput model.

        Args:
            vin: Raw input data from mempool API
            input_address_id: ID from input_addresses table

        Returns:
            TransactionInput Pydantic model
        """
        # Normalize and validate transaction ID
        prev_tx_id = TransactionProcessor.txid_to_hex(vin['txid'])

        # Detect and normalize script type
        script_type = TransactionProcessor.detect_script_type(
            vin['prevout']['scriptpubkey_type']
        )

        # Create TransactionInput Pydantic model
        return TransactionInput(
            prev_tx_id=prev_tx_id,
            prev_vout_index=vin['vout'],
            script_pubkey_type=ScriptPubkeyType(script_type),
            script_pubkey_address=vin['prevout']['scriptpubkey_address'],
            input_value_satoshi=vin['prevout']['value'],
            is_coinbase=vin.get('is_coinbase', False),
            input_address_id=input_address_id
        )

    @staticmethod
    def process_output(vout: Dict[str, Any], vout_index: int, output_address_id: int) -> TransactionOutput:
        #TODO change the function according to the style of the DataCollector
        #TODO as the input_address_id is generated after an input is proccesed 
        #TODO may be pass the DatabaseConnection class into the funciton 
        """
        Process transaction output into TransactionOutput model.

        Args:
            vout: Raw output data from mempool API
            vout_index: Index of this output in the transaction
            output_address_id: ID from output_addresses table

        Returns:
            TransactionOutput Pydantic model
        """
        # Detect and normalize script type
        script_type = TransactionProcessor.detect_script_type(
            vout['scriptpubkey_type']
        )

        # Create TransactionOutput Pydantic model
        return TransactionOutput(
            vout_index=vout_index,
            script_pubkey_type=ScriptPubkeyType(script_type),
            script_pubkey_address=vout['scriptpubkey_address'],
            output_value_satoshi=vout['value'],
            output_address_id=output_address_id
        )

    @staticmethod
    def process_transaction(
        tx_raw: Dict[str, Any],
        coordinator_id: int,
        db_conn: DatabaseConnection
    ) -> CoinjoinTransaction:
        """
        Process raw transaction data into CoinjoinTransaction model.

        This method orchestrates the full processing pipeline:
        1. Extract transaction metadata
        2. Process all inputs and outputs
        3. Update address tables
        4. Calculate derived fields (fees, rates)

        Args:
            tx_raw: Raw transaction data from mempool API
            coordinator_id: Coordinator ID from database
            db_conn: Database connection for address operations

        Returns:
            CoinjoinTransaction Pydantic model ready for insertion
        """
        # Step 1: Extract and normalize basic transaction info
        txid = TransactionProcessor.txid_to_hex(tx_raw['txid'])

        # Step 2: Process inputs - update/insert addresses and create TransactionInput objects
        processed_inputs = []
        total_input_value = 0

        for vin in tx_raw['vin']:
            # Extract input address info
            script_type = TransactionProcessor.detect_script_type(
                vin['prevout']['scriptpubkey_type']
            )
            address = vin['prevout']['scriptpubkey_address']
            value = vin['prevout']['value']

            # Update or insert input address and get ID
            input_address = db_conn.update_or_insert_input_address(
                address=address,
                script_type=script_type,
                value=value
            )

            # Process input into TransactionInput model
            tx_input = TransactionProcessor.process_input(
                vin=vin,
                input_address_id=input_address.input_address_id
            )

            processed_inputs.append(tx_input)
            total_input_value += value

        # Step 3: Process outputs - update/insert addresses and create TransactionOutput objects
        processed_outputs = []
        total_output_value = 0

        for vout_index, vout in enumerate(tx_raw['vout']):
            # Extract output address info
            script_type = TransactionProcessor.detect_script_type(
                vout['scriptpubkey_type']
            )
            address = vout['scriptpubkey_address']
            value = vout['value']

            # Update or insert output address and get ID
            output_address = db_conn.update_or_insert_output_address(
                address=address,
                script_type=script_type,
                value=value
            )

            # Process output into TransactionOutput model
            tx_output = TransactionProcessor.process_output(
                vout=vout,
                vout_index=vout_index,
                output_address_id=output_address.output_address_id
            )

            processed_outputs.append(tx_output)
            total_output_value += value

        # Step 4: Extract fee and calculate fee rate
        transaction_fee = tx_raw['fee']

        # Calculate fee rate (sat/vByte)
        # Virtual size (vsize) = weight / 4
        weight = tx_raw['weight']
        vsize = weight / 4
        fee_rate_sat_per_vbyte = transaction_fee / vsize if vsize > 0 else 0.0

        # Step 5: Extract block info and timestamp
        from datetime import datetime
        block_number = tx_raw['status']['block_height']
        block_time = datetime.fromtimestamp(tx_raw['status']['block_time'])

        # Step 6: Create and return CoinjoinTransaction model
        return CoinjoinTransaction(
            tx_id=txid,
            number_inputs=len(processed_inputs),
            number_outputs=len(processed_outputs),
            value_inputs=total_input_value,
            value_outputs=total_output_value,
            inputs=processed_inputs,
            outputs=processed_outputs,
            coordinator_id=coordinator_id,
            transaction_fee=transaction_fee,
            block_number=block_number,
            block_time=block_time,
            raw_size_bytes=tx_raw['size'],
            weight=weight,
            fee_rate_sat_per_vbyte=fee_rate_sat_per_vbyte,
            processed=False  # Not yet processed for clustering
        )

    @staticmethod
    def is_coinjoin_transaction(tx_data: Dict[str, Any]) -> bool:
        """
        Determine if a transaction is a CoinJoin transaction using heuristics.

        Common CoinJoin patterns:
        - Multiple inputs (typically >= 2)
        - Multiple outputs (typically >= 3)
        - Equal-valued outputs (Wasabi, Whirlpool pattern)
        - Specific output value distributions

        Args:
            tx_data: Raw transaction data from API

        Returns:
            True if transaction appears to be a CoinJoin
        """
        # TODO: Implement CoinJoin detection heuristics (currently no need)
        # Simple heuristic to start:
        # - Check if there are at least 2 equal-valued outputs
        # - Check if number of inputs >= 2 and outputs >= 3
        pass