"""
Models package for CoinJoin parser.

Exports Pydantic models for CoinJoin transaction data.
"""

from models.coinjoin_data import (
    # Enums
    ScriptPubkeyType,

    # Struct models
    TransactionInput,
    TransactionOutput,

    # Table models
    Coordinator,
    InputAddress,
    OutputAddress,
    CoinjoinTransaction,
)

from models.database_connection import (
    # Database connection
    DatabaseConnection,
)

from models.transaction_processor import (
    # Transaction processing
    TransactionProcessor,
)

__all__ = [
    # Enums
    "ScriptPubkeyType",

    # Struct models
    "TransactionInput",
    "TransactionOutput",

    # Table models
    "Coordinator",
    "InputAddress",
    "OutputAddress",
    "CoinjoinTransaction",

    # Database connection
    "DatabaseConnection",

    # Transaction processing
    "TransactionProcessor",
]
