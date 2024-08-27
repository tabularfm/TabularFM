# Gittables datasets

1. Download the csv file version [here](https://zenodo.org/record/6515973) of [Gittables](https://gittables.github.io/)
2. Run the preprocessing as follows
    * Process level 1: `python shortlist_gittables_lv1.py`
    * Process level 2: `python shortlist_gittables_lv2.py`
    * Process level 3: `python shortlist_gittables_lv3.py`
    * Merge dataset: `python merge_gittables.py`
3. Clean and generate metdata: `python preprocess_data_gittables.py`

# Strategy to process Gittables data

* LEVEL 1
    * Shortlist tables with #rows > 1st quartile of rows (Q1 rows) and #cols >= Q1 cols
* LEVEL 2
    * <s>Avoid tables with long texts in column names or values (e.g. FUL_1.csv)</s>
    * Avoid tables with large NA column values
    * [Not-applicable] Split special characters away from numbers 
    * [Not-applicable] *(Avoid tables with meaningless column names (e.g. BooldyMary))*
* LEVEL 3
    * Find prefixes and set group for datasets with the same prefixes
        * Group datasets having same prefixes
        * Group single datasets

* MERGE
    * Merge datasets within a same group into a single csv file

* Then apply processing script in the repo to further shortlist the datasets