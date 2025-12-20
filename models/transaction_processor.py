"""
CoinJoin Transaction Processor

This module processes CoinJoin transaction data and prepares it for database storage.
"""

from typing import Dict, Any, TYPE_CHECKING
from datetime import datetime

# Import Pydantic models
from models.coinjoin_data import (
    ScriptPubkeyType,
    TransactionInput,
    TransactionOutput,
    CoinjoinTransaction,
)

if TYPE_CHECKING:
    from models.database_connection import DatabaseConnection

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
        db_conn: "DatabaseConnection"
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

            # Check if input address is also used as output and update flag
            used_as_output = db_conn.check_input_is_also_output_update_output(address)

            # Update or insert input address and get ID
            input_address = db_conn.update_or_insert_input_address(
                address=address,
                script_type=script_type,
                value=value,
                used_as_output=used_as_output
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

            # Check if output address is also used as input and update flag
            used_as_input = db_conn.check_output_is_also_input_update_input(address)

            # Update or insert output address and get ID
            output_address = db_conn.update_or_insert_output_address(
                address=address,
                script_type=script_type,
                value=value,
                used_as_input=used_as_input
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
