"""Tab-separated input/output. Every competition file is a .tsv; quoting is disabled because names and addresses contain quotes."""
import polars as pl


def read_tsv(path, columns=None):
    """Read a competition .tsv with every column as a string."""
    return pl.read_csv(path, separator="\t", quote_char=None, infer_schema=False, columns=columns)


def write_tsv(df, path):
    df.write_csv(path, separator="\t", quote_style="never")


def explode_ids(df, id_col, list_col, out_col="entity_id"):
    """(source1_entity_id, "id1,id2,...") rows -> one row per (source1_entity_id, id); empty lists give no rows."""
    return (df.select(id_col, pl.col(list_col).fill_null("").str.split(",").alias(out_col))
              .explode(out_col).filter(pl.col(out_col) != ""))
