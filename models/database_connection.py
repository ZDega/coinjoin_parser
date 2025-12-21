"""
Database Connection Module

This module provides the DatabaseConnection class for managing DuckDB database
connections and operations for CoinJoin transaction data.
"""

import os
from typing import Dict, Any, Optional
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
            self.conn.execute("SET threads=1")
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
            """SELECT input_address_id, address, used_as_output, script_type,
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
            """SELECT output_address_id, address, used_as_input, script_type,
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

    def update_or_insert_input_address(self, address: str, script_type: str, value: int, used_as_output: bool) -> InputAddress:
        """
        Update existing input address statistics or create new entry if it doesn't exist.

        This method increments usage counters and updates total amounts for addresses
        used as inputs in CoinJoin transactions.

        Args:
            address: Script pubkey address (Bitcoin address format)
            script_type: Script pubkey type (p2pkh, p2sh, v0_p2wpkh, etc.)
            value: Input value in satoshis to add to total_amount_spent_in_cj
            used_as_output: Boolean indicating if this address has been used as an output address

        Returns:
            InputAddress object with updated values
        """
        # Check if input address already exists
        result = self.get_input_address(address, script_type)
        if result and result.input_address_id == 1:
            print(result)

        if result:
            # Address exists, update statistics in database
            self.conn.execute(
                """UPDATE input_addresses
                   SET number_of_cjs_used_in_as_input = number_of_cjs_used_in_as_input + 1,
                        total_amount_spent_in_cj = total_amount_spent_in_cj + ?,
                        used_as_output = ?
                   WHERE input_address_id = ?""",
                [value, used_as_output, result.input_address_id]
            )
            print(result.number_of_cjs_used_in_as_input + 1)
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
               VALUES (?, ?, ?::script_pubkey_type, 1, ?)""",
            [address, used_as_output, script_type, value]
        )

        # Get the newly created input address with full data
        result = self.conn.execute(
            """SELECT input_address_id, address, used_as_output, script_type,
                      number_of_cjs_used_in_as_input, total_amount_spent_in_cj
               FROM input_addresses
               WHERE address = ?""",
            [address]
        ).fetchone()

        #print(f"    ✅ Created new input address with ID: {result[0]}")
        return InputAddress.from_db_row(*result)

    def update_or_insert_output_address(self, address: str, script_type: str, value: int, used_as_input: bool) -> OutputAddress:
        """
        Update existing output address statistics or create new entry if it doesn't exist.

        This method increments usage counters and updates total amounts for addresses
        used as outputs in CoinJoin transactions.

        Args:
            address: Script pubkey address (Bitcoin address format)
            script_type: Script pubkey type (p2pkh, p2sh, v0_p2wpkh, etc.)
            value: Output value in satoshis to add to total_amount_received_in_cj
            used_as_input: Boolean indicating if this address has been used as an input address

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
                    total_amount_received_in_cj = total_amount_received_in_cj + ?,
                    used_as_input = ?
                   WHERE output_address_id = ?""",
                [value, used_as_input, result.output_address_id]
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
               VALUES (?, ?, ?::script_pubkey_type, 1, ?)""",
            [address, used_as_input, script_type, value]
        )

        # Get the newly created output address with full data
        result = self.conn.execute(
            """SELECT output_address_id, address, used_as_input, script_type,
                      number_of_cjs_used_in_as_output, total_amount_received_in_cj
               FROM output_addresses
               WHERE address = ?""",
            [address]
        ).fetchone()

        #print(f"    ✅ Created new output address with ID: {result[0]}")
        return OutputAddress.from_db_row(*result)
    
    def check_input_is_also_output_update_output(self, address: str) -> bool:
        """
        Check if an input address is also used as an output address,
        and update the used_as_input flag in output_addresses table if necessary.

        Args:
            address: Script pubkey address (Bitcoin address format)
        Returns:
            True if the address is found and updated, False otherwise
            Also updates the used_as_input flag in output_addresses table if necessary.
        """
        # Check if the address exists in output_addresses
        result = self.conn.execute(
            """SELECT output_address_id, used_as_input
               FROM output_addresses
               WHERE address = ?""",
            [address]
        ).fetchone()

        if result:
            output_address_id, used_as_input = result
            if not used_as_input:
                # Update the used_as_input flag to True
                self.conn.execute(
                    """UPDATE output_addresses
                       SET used_as_input = TRUE
                       WHERE output_address_id = ?""",
                    [output_address_id]
                )
                print(f"    🔄 Updated output address {address[:16]}... (ID: {output_address_id}) to mark as used as input")
            return True

        return False
    
    def check_output_is_also_input_update_input(self, address: str) -> bool:
        """
        Check if an output address is also used as an input address,
        and update the used_as_output flag in input_addresses table if necessary.

        Args:
            address: Script pubkey address (Bitcoin address format)
        Returns:
            True if the address is found and updated, False otherwise
            Also updates the used_as_output flag in input_addresses table if necessary.
        """
        # Check if the address exists in input_addresses
        result = self.conn.execute(
            """SELECT input_address_id, used_as_output
               FROM input_addresses
               WHERE address = ?""",
            [address]
        ).fetchone()

        if result:
            input_address_id, used_as_output = result
            if not used_as_output:
                # Update the used_as_output flag to True
                self.conn.execute(
                    """UPDATE input_addresses
                       SET used_as_output = TRUE
                       WHERE input_address_id = ?""",
                    [input_address_id]
                )
                print(f"    🔄 Updated input address {address[:16]}... (ID: {input_address_id}) to mark as used as output")
            return True

        return False
    
    def get_coinjoin_transaction(self, tx_id: str, show_progress: Optional[bool] = False) -> Optional[CoinjoinTransaction]:
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


