"""
Pydantic models for CoinJoin transaction data.

These models correspond to the DuckDB schema defined in data/create_data_base.py
"""

from typing import List, Optional, Dict, Any
from datetime import datetime
from enum import Enum
from pydantic import BaseModel, Field, field_validator
from tqdm import tqdm


# ============================================================================
# ENUMS
# ============================================================================

class ScriptPubkeyType(str, Enum):
    """Script pubkey types matching DuckDB script_pubkey_type enum."""
    P2PKH = "p2pkh"
    P2SH = "p2sh"
    V0_P2WPKH = "v0_p2wpkh"
    V0_P2WSH = "v0_p2wsh"
    V1_P2TR = "v1_p2tr"
    OTHER = "other"


# ============================================================================
# STRUCT MODELS (for nested transaction data)
# ============================================================================

class TransactionInput(BaseModel):
    """
    Transaction input struct matching DuckDB transaction_input type.

    Corresponds to:
        CREATE TYPE IF NOT EXISTS transaction_input AS STRUCT (
            prev_tx_id BLOB,
            prev_vout_index INTEGER,
            script_pubkey_type script_pubkey_type,
            script_pubkey_address BLOB,
            input_value_satoshi BIGINT,
            is_coinbase BOOL,
            input_address_id INTEGER
        )
    """
    prev_tx_id: str = Field(..., description="Previous transaction ID (32 bytes)")
    prev_vout_index: int = Field(..., description="Previous output index", ge=0)
    script_pubkey_type: ScriptPubkeyType = Field(..., description="Script pubkey type")
    script_pubkey_address: str = Field(..., description="Script pubkey address in binary form")
    input_value_satoshi: int = Field(..., description="Input value in satoshis", ge=0)
    is_coinbase: bool = Field(default=False, description="Whether this is a coinbase input")
    input_address_id: int = Field(..., description="Foreign key to input_addresses table", ge=1)

    @classmethod
    def from_db_row(cls,
                    prev_tx_id: str,
                    prev_vout_index: int,
                    script_pubkey_type: str,
                    script_pubkey_address: str,
                    input_value_satoshi: int,
                    is_coinbase: bool,
                    input_address_id: int) -> "TransactionInput":
        """
        Create TransactionInput from database query result or struct.

        Args:
            prev_tx_id: Previous transaction ID in hex format
            prev_vout_index: Previous output index
            script_pubkey_type: Script pubkey type string
            script_pubkey_address: Script pubkey address in hex format
            input_value_satoshi: Input value in satoshis
            is_coinbase: Whether this is a coinbase input
            input_address_id: Foreign key to input_addresses table

        Returns:
            TransactionInput instance
        """
        return cls(
            prev_tx_id=prev_tx_id,
            prev_vout_index=prev_vout_index,
            script_pubkey_type=ScriptPubkeyType(script_pubkey_type),
            script_pubkey_address=script_pubkey_address,
            input_value_satoshi=input_value_satoshi,
            is_coinbase=is_coinbase,
            input_address_id=input_address_id
        )

    @classmethod
    def from_db_rows(cls,
                     rows: List[Dict[str, Any]],
                     show_progress: bool = False,
                     desc: str = "Processing inputs") -> List["TransactionInput"]:
        """
        Create list of TransactionInput objects from database rows with optional progress bar.

        Args:
            rows: List of dicts/structs from DuckDB (each representing an input)
            show_progress: Whether to show progress bar (default: False)
            desc: Description for progress bar (default: "Processing inputs")

        Returns:
            List of TransactionInput instances

        Example:
            inputs = TransactionInput.from_db_rows(result['inputs'], show_progress=True)
        """
        iterator = tqdm(rows, desc=desc) if show_progress else rows
        return [
            cls.from_db_row(
                prev_tx_id=row['prev_tx_id'],
                prev_vout_index=row['prev_vout_index'],
                script_pubkey_type=row['script_pubkey_type'],
                script_pubkey_address=row['script_pubkey_address'],
                input_value_satoshi=row['input_value_satoshi'],
                is_coinbase=row.get('is_coinbase', False),
                input_address_id=row['input_address_id']
            )
            for row in iterator
        ]

    @staticmethod
    def from_object_to_db(input: "TransactionInput") -> Dict[str, Any]:
        """
        Convert TransactionInput Pydantic model to DuckDB struct format.

        This method prepares a TransactionInput object for insertion into DuckDB
        by converting hex strings to bytes and extracting enum values.

        Args:
            input: TransactionInput Pydantic model instance

        Returns:
            Dictionary with DuckDB-compatible struct fields (bytes for BLOB fields)

        Example:
            input_struct = TransactionInput.from_object_to_db(input_obj)
            # Use in INSERT statement as part of transaction_input[] array
        """
        return {
            "prev_tx_id": bytes.fromhex(input.prev_tx_id),
            "prev_vout_index": input.prev_vout_index,
            "script_pubkey_type": input.script_pubkey_type.value,
            "script_pubkey_address": bytes.fromhex(input.script_pubkey_address),
            "input_value_satoshi": input.input_value_satoshi,
            "is_coinbase": input.is_coinbase,
            "input_address_id": input.input_address_id
        }
    class Config:
        use_enum_values = True


class TransactionOutput(BaseModel):
    """
    Transaction output struct matching DuckDB transaction_output type.

    Corresponds to:
        CREATE TYPE IF NOT EXISTS transaction_output AS STRUCT (
            vout_index INTEGER,
            script_pubkey_type script_pubkey_type,
            script_pubkey_address BLOB,
            output_value_satoshi BIGINT,
            output_address_id INTEGER
        )
    """
    vout_index: int = Field(..., description="Output index in transaction", ge=0)
    script_pubkey_type: ScriptPubkeyType = Field(..., description="Script pubkey type")
    script_pubkey_address: str = Field(..., description="Script pubkey address in binary form")
    output_value_satoshi: int = Field(..., description="Output value in satoshis", ge=0)
    output_address_id: int = Field(..., description="Foreign key to output_addresses table", ge=1)

    @classmethod
    def from_db_row(cls,
                    vout_index: int,
                    script_pubkey_type: str,
                    script_pubkey_address: str,
                    output_value_satoshi: int,
                    output_address_id: int) -> "TransactionOutput":
        """
        Create TransactionOutput from database query result or struct.

        Args:
            vout_index: Output index in transaction
            script_pubkey_type: Script pubkey type string
            script_pubkey_address: Script pubkey address in hex format
            output_value_satoshi: Output value in satoshis
            output_address_id: Foreign key to output_addresses table

        Returns:
            TransactionOutput instance
        """
        return cls(
            vout_index=vout_index,
            script_pubkey_type=ScriptPubkeyType(script_pubkey_type),
            script_pubkey_address=script_pubkey_address,
            output_value_satoshi=output_value_satoshi,
            output_address_id=output_address_id
        )
    

    @classmethod
    def from_db_rows(cls,
                     rows: List[Dict[str, Any]],
                     show_progress: bool = False,
                     desc: str = "Processing outputs") -> List["TransactionOutput"]:
        """
        Create list of TransactionOutput objects from database rows with optional progress bar.

        Args:
            rows: List of dicts/structs from DuckDB (each representing an output)
            show_progress: Whether to show progress bar (default: False)
            desc: Description for progress bar (default: "Processing outputs")

        Returns:
            List of TransactionOutput instances

        Example:
            outputs = TransactionOutput.from_db_rows(result['outputs'], show_progress=True)
        """
        iterator = tqdm(rows, desc=desc) if show_progress else rows
        return [
            cls.from_db_row(
                vout_index=row['vout_index'],
                script_pubkey_type=row['script_pubkey_type'],
                script_pubkey_address=row['script_pubkey_address'],
                output_value_satoshi=row['output_value_satoshi'],
                output_address_id=row['output_address_id']
            )
            for row in iterator
        ]


    @staticmethod
    def from_object_to_db(output: "TransactionOutput") -> Dict[str, Any]:
        """
        Convert TransactionOutput Pydantic model to DuckDB struct format.

        This method prepares a TransactionOutput object for insertion into DuckDB
        by converting hex strings to bytes and extracting enum values.

        Args:
            output: TransactionOutput Pydantic model instance

        Returns:
            Dictionary with DuckDB-compatible struct fields (bytes for BLOB fields)

        Example:
            output_struct = TransactionOutput.from_object_to_db(output_obj)
            # Use in INSERT statement as part of transaction_output[] array
        """
        return {
            "vout_index": output.vout_index,
            "script_pubkey_type": output.script_pubkey_type,
            "script_pubkey_address": bytes.fromhex(output.script_pubkey_address),
            "output_value_satoshi": output.output_value_satoshi,
            "output_address_id": output.output_address_id
        }
    class Config:
        use_enum_values = True


# ============================================================================
# TABLE MODELS
# ============================================================================

class Coordinator(BaseModel):
    """
    Coordinator model matching DuckDB coordinators table.

    Corresponds to:
        CREATE TABLE IF NOT EXISTS coordinators (
            coor_id INTEGER PRIMARY KEY DEFAULT nextval('coor_id_seq'),
            coordinator_endpoint VARCHAR UNIQUE
        )
    """
    coor_id: Optional[int] = Field(None, description="Internal numeric ID (auto-generated)", ge=1)
    coordinator_endpoint: str = Field(..., description="URL of coordinator", min_length=1)

    @classmethod
    def from_db_row(cls, coor_id: int, coordinator_endpoint: str) -> "Coordinator":
        """
        Create Coordinator from database query result.

        Args:
            coor_id: Coordinator ID
            coordinator_endpoint: Coordinator endpoint URL

        Returns:
            Coordinator instance
        """
        return cls(
            coor_id=coor_id,
            coordinator_endpoint=coordinator_endpoint
        )

    class Config:
        from_attributes = True


class InputAddress(BaseModel):
    """
    Input address model matching DuckDB input_addresses table.

    Corresponds to:
        CREATE TABLE IF NOT EXISTS input_addresses (
            input_address_id INTEGER PRIMARY KEY DEFAULT nextval('input_id_seq'),
            address BLOB UNIQUE,
            used_as_output BOOLEAN,
            script_type script_pubkey_type,
            number_of_cjs_used_in_as_input INTEGER,
            total_amount_spent_in_cj BIGINT
        )
    """
    input_address_id: Optional[int] = Field(None, description="Primary key (auto-generated)", ge=1)
    address: str = Field(..., description="Script pubkey address in binary form")
    used_as_output: bool = Field(..., description="True if this address also appears as CJ output")
    script_type: ScriptPubkeyType = Field(..., description="Script pubkey type")
    number_of_cjs_used_in_as_input: int = Field(
        default=0,
        description="Number of CoinJoin transactions this address was used as input",
        ge=0
    )
    total_amount_spent_in_cj: int = Field(
        default=0,
        description="Total amount spent in CoinJoin transactions (satoshis)",
        ge=0
    )

    @classmethod
    def from_db_row(cls, input_address_id: int,
                         address: str,
                         used_as_output: bool,
                         script_type: str,
                         number_of_cjs_in_as_input: int,
                         total_amount_spend_in_cj: int) -> "InputAddress":
        """
        Create InputAddress from database query result.

        Args:
            row: Tuple from query (input_address_id, address_hex, used_as_output,
                 script_type, number_of_cjs_used_in_as_input, total_amount_spent_in_cj)

        Returns:
            InputAddress instance
        """
        return cls(
            input_address_id=input_address_id,
            address=address,
            used_as_output=used_as_output,
            script_type=ScriptPubkeyType(script_type),
            number_of_cjs_used_in_as_input=number_of_cjs_in_as_input,
            total_amount_spent_in_cj=total_amount_spend_in_cj
        )

    class Config:
        from_attributes = True
        use_enum_values = True


class OutputAddress(BaseModel):
    """
    Output address model matching DuckDB output_addresses table.

    Corresponds to:
        CREATE TABLE IF NOT EXISTS output_addresses (
            output_address_id INTEGER PRIMARY KEY DEFAULT nextval('output_id_seq'),
            address BLOB UNIQUE,
            used_as_input BOOLEAN,
            script_type script_pubkey_type,
            number_of_cjs_used_in_as_output INTEGER,
            total_amount_received_in_cj BIGINT
        )
    """
    output_address_id: Optional[int] = Field(None, description="Primary key (auto-generated)", ge=1)
    address: str = Field(..., description="Script pubkey address in binary form")
    used_as_input: bool = Field(..., description="True if this address also appears as CJ input")
    script_type: ScriptPubkeyType = Field(..., description="Script pubkey type")
    number_of_cjs_used_in_as_output: int = Field(
        default=0,
        description="Number of CoinJoin transactions this address was used as output",
        ge=0
    )
    total_amount_received_in_cj: int = Field(
        default=0,
        description="Total amount received in CoinJoin transactions (satoshis)",
        ge=0
    )

    @classmethod
    def from_db_row(cls, output_address_id: int,
                         address: str,
                         used_as_input: bool,
                         script_type: str,
                         number_of_cjs_in_as_output: int,
                         total_amount_received_in_cj: int) -> "OutputAddress":
        """
        Create InputAddress from database query result.

        Args:
            row: Tuple from query (input_address_id, address_hex, used_as_output,
                 script_type, number_of_cjs_used_in_as_input, total_amount_spent_in_cj)

        Returns:
            InputAddress instance
        """
        return cls(
            output_address_id=output_address_id,
            address=address,
            used_as_input=used_as_input,
            script_type=ScriptPubkeyType(script_type),
            number_of_cjs_in_as_output=number_of_cjs_in_as_output,
            total_amount_received_in_cj=total_amount_received_in_cj
        )

    class Config:
        from_attributes = True
        use_enum_values = True


class CoinjoinTransaction(BaseModel):
    """
    CoinJoin transaction model matching DuckDB coinjoin_transactions table.

    Corresponds to:
        CREATE TABLE IF NOT EXISTS coinjoin_transactions (
            tx_id_int INTEGER PRIMARY KEY DEFAULT nextval('tx_id_seq'),
            tx_id BLOB UNIQUE,
            number_inputs INTEGER,
            number_outputs INTEGER,
            value_inputs BIGINT,
            value_outputs BIGINT,
            inputs transaction_input[],
            outputs transaction_output[],
            coordinator_id INTEGER REFERENCES coordinators(coor_id),
            transaction_fee BIGINT,
            block_number INTEGER,
            block_time TIMESTAMP,
            raw_size_bytes INTEGER,
            weight INTEGER,
            fee_rate_sat_per_vbyte DOUBLE,
            processed BOOLEAN NOT NULL DEFAULT FALSE
        )
    """
    tx_id_int: Optional[int] = Field(None, description="Internal primary key (auto-generated)", ge=1)
    tx_id: str = Field(..., description="Raw 32-byte transaction ID")

    number_inputs: int = Field(..., description="Number of transaction inputs", ge=1)
    number_outputs: int = Field(..., description="Number of transaction outputs", ge=1)
    value_inputs: int = Field(..., description="Sum of input values in satoshis", ge=0)
    value_outputs: int = Field(..., description="Sum of output values in satoshis", ge=0)

    inputs: List[TransactionInput] = Field(..., description="Full list of transaction inputs")
    outputs: List[TransactionOutput] = Field(..., description="Full list of transaction outputs")

    coordinator_id: int = Field(..., description="Foreign key to coordinators table", ge=1)

    transaction_fee: int = Field(..., description="Transaction fee in satoshis", ge=0)
    block_number: int = Field(..., description="Block height", ge=0)
    block_time: datetime = Field(..., description="Block timestamp")

    raw_size_bytes: int = Field(..., description="Serialized transaction size in bytes", ge=0)
    weight: int = Field(..., description="Transaction weight units", ge=0)
    fee_rate_sat_per_vbyte: float = Field(..., description="Fee rate in sat/vByte", ge=0.0)

    processed: bool = Field(default=False, description="Whether transaction has been processed for clustering")

    @field_validator('inputs')
    @classmethod
    def validate_inputs_count(cls, v, info):
        """Validate that inputs list matches number_inputs."""
        if 'number_inputs' in info.data and len(v) != info.data['number_inputs']:
            raise ValueError(f"inputs length ({len(v)}) must match number_inputs ({info.data['number_inputs']})")
        return v

    @field_validator('outputs')
    @classmethod
    def validate_outputs_count(cls, v, info):
        """Validate that outputs list matches number_outputs."""
        if 'number_outputs' in info.data and len(v) != info.data['number_outputs']:
            raise ValueError(f"outputs length ({len(v)}) must match number_outputs ({info.data['number_outputs']})")
        return v

    @field_validator('transaction_fee')
    @classmethod
    def validate_fee(cls, v, info):
        """Validate that transaction fee equals inputs - outputs."""
        if 'value_inputs' in info.data and 'value_outputs' in info.data:
            expected_fee = info.data['value_inputs'] - info.data['value_outputs']
            if v != expected_fee:
                raise ValueError(
                    f"transaction_fee ({v}) must equal value_inputs - value_outputs ({expected_fee})"
                )
        return v

    @classmethod
    def from_db_row(cls,
                    tx_id_int: int,
                    tx_id: str,
                    number_inputs: int,
                    number_outputs: int,
                    value_inputs: int,
                    value_outputs: int,
                    inputs: List[TransactionInput],
                    outputs: List[TransactionOutput],
                    coordinator_id: int,
                    transaction_fee: int,
                    block_number: int,
                    block_time: datetime,
                    raw_size_bytes: int,
                    weight: int,
                    fee_rate_sat_per_vbyte: float,
                    processed: bool) -> "CoinjoinTransaction":
        """
        Create CoinjoinTransaction from database query result.

        Args:
            tx_id_int: Internal transaction ID
            tx_id: Transaction ID in hex format
            number_inputs: Number of inputs
            number_outputs: Number of outputs
            value_inputs: Total input value in satoshis
            value_outputs: Total output value in satoshis
            inputs: List of TransactionInput objects
            outputs: List of TransactionOutput objects
            coordinator_id: Coordinator ID
            transaction_fee: Transaction fee in satoshis
            block_number: Block height
            block_time: Block timestamp
            raw_size_bytes: Transaction size in bytes
            weight: Transaction weight
            fee_rate_sat_per_vbyte: Fee rate in sat/vByte
            processed: Whether transaction has been processed

        Returns:
            CoinjoinTransaction instance
        """
        return cls(
            tx_id_int=tx_id_int,
            tx_id=tx_id,
            number_inputs=number_inputs,
            number_outputs=number_outputs,
            value_inputs=value_inputs,
            value_outputs=value_outputs,
            inputs=inputs,
            outputs=outputs,
            coordinator_id=coordinator_id,
            transaction_fee=transaction_fee,
            block_number=block_number,
            block_time=block_time,
            raw_size_bytes=raw_size_bytes,
            weight=weight,
            fee_rate_sat_per_vbyte=fee_rate_sat_per_vbyte,
            processed=processed
        )

    #TODO check correctness
    @staticmethod
    def from_object_to_db(coinjoin_transaction: "CoinjoinTransaction") -> Dict[str, Any]:
        """
        Convert CoinjoinTransaction Pydantic model to DuckDB-compatible format for insertion.

        This method prepares a complete CoinJoin transaction for database insertion by:
        1. Converting hex transaction ID to bytes
        2. Converting nested TransactionInput list to struct array
        3. Converting nested TransactionOutput list to struct array
        4. Preparing all other fields for insertion

        Args:
            coinjoin_transaction: CoinjoinTransaction Pydantic model instance

        Returns:
            Dictionary with all fields ready for DuckDB INSERT statement

        Example:
            tx_data = CoinjoinTransaction.from_object_to_db(transaction)
            # Use in insert_coinjoin_transaction method
        """
        # Convert input objects to DuckDB struct format
        inputs_structs = [
            TransactionInput.from_object_to_db(inp)
            for inp in coinjoin_transaction.inputs
        ]

        # Convert output objects to DuckDB struct format
        outputs_structs = [
            TransactionOutput.from_object_to_db(out)
            for out in coinjoin_transaction.outputs
        ]

        return {
            "tx_id": bytes.fromhex(coinjoin_transaction.tx_id),
            "number_inputs": coinjoin_transaction.number_inputs,
            "number_outputs": coinjoin_transaction.number_outputs,
            "value_inputs": coinjoin_transaction.value_inputs,
            "value_outputs": coinjoin_transaction.value_outputs,
            "inputs": inputs_structs,
            "outputs": outputs_structs,
            "coordinator_id": coinjoin_transaction.coordinator_id,
            "transaction_fee": coinjoin_transaction.transaction_fee,
            "block_number": coinjoin_transaction.block_number,
            "block_time": coinjoin_transaction.block_time,
            "raw_size_bytes": coinjoin_transaction.raw_size_bytes,
            "weight": coinjoin_transaction.weight,
            "fee_rate_sat_per_vbyte": coinjoin_transaction.fee_rate_sat_per_vbyte,
            "processed": coinjoin_transaction.processed
        }
    class Config:
        from_attributes = True
        use_enum_values = True
