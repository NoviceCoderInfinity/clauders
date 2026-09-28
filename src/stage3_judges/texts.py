"""Pair texts shared by the judges and the combiners: "business_name | business_address" (name alone when the address is empty)."""
import polars as pl


def text(name, address):
    """polars expression over the raw business_name / business_address columns."""
    a = address.fill_null("").str.strip_chars(); n = name.fill_null("").str.strip_chars()
    return pl.when(a != "").then(n + " | " + a).otherwise(n)
