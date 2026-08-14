"""composite matching: new_feature_matches join table + explanation column

Revision ID: c4d8e1f20a35
Revises: 9726524b1818
Create Date: 2026-08-09

Composite matching (requirements.md §3/§5): a requirement can match SEVERAL
existing features, so matches move into a join table. Also adds the exact-match
`explanation` column and drops the legacy single-match `matching_existing_feature_id`.
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'c4d8e1f20a35'
down_revision: Union[str, Sequence[str], None] = '9726524b1818'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    # join table: ONE requirement -> MANY matched existing features
    op.create_table(
        'new_feature_matches',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('new_feature_id', sa.Integer(), nullable=False),
        sa.Column('completed_feature_id', sa.Integer(), nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=True),
        sa.ForeignKeyConstraint(['new_feature_id'], ['new_features.id'], ondelete='CASCADE'),
        sa.ForeignKeyConstraint(['completed_feature_id'], ['completed_features.id']),
        sa.PrimaryKeyConstraint('id'),
    )
    op.create_index(
        'ix_new_feature_matches_new_feature_id',
        'new_feature_matches',
        ['new_feature_id'],
    )

    # exact-match explanation
    op.add_column('new_features', sa.Column('explanation', sa.Text(), nullable=True))

    # retire the legacy single-match column (join table replaces it)
    op.drop_column('new_features', 'matching_existing_feature_id')


def downgrade() -> None:
    """Downgrade schema."""
    op.add_column(
        'new_features',
        sa.Column('matching_existing_feature_id', sa.INTEGER(), autoincrement=False, nullable=True),
    )
    op.create_foreign_key(
        'new_features_matching_existing_feature_id_fkey',
        'new_features', 'completed_features',
        ['matching_existing_feature_id'], ['id'],
    )
    op.drop_column('new_features', 'explanation')

    op.drop_index('ix_new_feature_matches_new_feature_id', table_name='new_feature_matches')
    op.drop_table('new_feature_matches')
