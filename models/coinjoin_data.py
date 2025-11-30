"""
Pydantic models for CoinJoin transaction data.

These models correspond to the DuckDB schema defined in data/create_data_base.py
"""

from typing import List, Optional
from datetime import datetime
from enum import Enum
from pydantic import BaseModel, Field, field_validator


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

    class Config:
        from_attributes = True
        use_enum_values = True
