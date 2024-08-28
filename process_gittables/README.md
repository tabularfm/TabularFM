# Gittables datasets

1. Download the csv file version [here](https://zenodo.org/record/6515973) of [Gittables](https://gittables.github.io/)
2. Create the following directories: `data/gittables_lv2`, `data/gittables_lv3`, `dataset_gittables`, `eda_gittables`. Put your data inside the `dataset_gittables` folder, nested inside another folder name (for example, `dataset_gittables/tables1`, `dataset_gittables/tables2`)
3. In the root directory of the project, run the preprocessing as follows
   - Process level 1: `python process_gittables/shortlist_gittables_lv1.py`
   - Process level 2: `python process_gittables/shortlist_gittables_lv2.py`
   - Process level 3: `python process_gittables/shortlist_gittables_lv3.py`
   - Merge dataset: `python process_gittables/merge_gittables.py`
4. Clean and generate metdata: `python process_gittables/preprocess_data_gittables.py`

# Strategy to process Gittables data

- LEVEL 1
  - Shortlist tables with #rows > 1st quartile of rows (Q1 rows) and #cols >= Q1 cols
- LEVEL 2
  - <s>Avoid tables with long texts in column names or values (e.g. FUL_1.csv)</s>
  - Avoid tables with large NA column values
  - [Not-applicable] Split special characters away from numbers
  - [Not-applicable] _(Avoid tables with meaningless column names (e.g. BooldyMary))_
- LEVEL 3

  - Find prefixes and set group for datasets with the same prefixes
    - Group datasets having same prefixes
    - Group single datasets

- MERGE

  - Merge datasets within a same group into a single csv file

- Then apply processing script in the repo to further shortlist the datasets
