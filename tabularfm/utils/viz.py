import os
from pycirclize import Circos
from pycirclize.parser import Matrix
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
from wordcloud import WordCloud

def merge_column_shapes(ft_report, st_report, ds_name):
    df_shapes_ft = ft_report.get_details(property_name="Column Shapes").copy()
    df_shapes_st = st_report.get_details(property_name="Column Shapes").copy()

    df_shapes_ft.rename(columns={'Score': 'ft_score'}, inplace=True)
    df_shapes_ft.drop(columns=['Metric'], inplace=True)
    df_shapes_st.rename(columns={'Score': 'st_score'}, inplace=True)
    df_shapes_st.drop(columns=['Metric'], inplace=True)

    df_shapes = df_shapes_ft.merge(df_shapes_st, on='Column')
    df_shapes['dataset'] = ds_name
    
    return df_shapes


def visualize_colshape(model_type, split_set, df_shapes, save_path):
    # clean columns
    if 'Error_x' in df_shapes.columns or 'Error_y' in df_shapes.columns:
        na_indices = df_shapes[~df_shapes['Error_x'].isna() | (~df_shapes['Error_y'].isna())].index

        df_shapes.drop(index=na_indices, inplace=True)
        df_shapes.drop(columns=['Error_x', 'Error_y'], inplace=True)
        
    df_shapes['ft_st'] = df_shapes['ft_score'] - df_shapes['st_score']
    df_shapes['st_ft'] = df_shapes['st_score'] - df_shapes['ft_score']
    # df_shapes.sort_values(by=['diff'], ascending=False, inplace=True)
    
    # standardize column data
    df_shapes.replace('\n',' ', regex=True, inplace=True)
    
    # Remove columns with <string><number> or only <number>
    rules = r'Atr([0-9]*)(.*)|^([\s\d\d+\.\d+]+)$|Unnamed'
    filter = df_shapes['Column'].str.contains(rules)
    df_shapes = df_shapes[~filter]
    
    df_shapes.to_csv(os.path.join(save_path, f'colshape_{model_type}_{split_set}.csv'))

    col_names = df_shapes['Column'].to_list()
    freqs_ft = df_shapes['ft_st'].to_list()
    freqs_st = df_shapes['st_ft'].to_list()

    rs_dict_ft = {col_names[i]: freqs_ft[i] for i in range(len(col_names))}
    rs_dict_st = {col_names[i]: freqs_st[i] for i in range(len(col_names))}

    # FT > ST
    wordcloud_best = WordCloud(max_font_size=50, max_words=len(rs_dict_ft), background_color="white",
                        width=1280, height=800).generate_from_frequencies(rs_dict_ft)

    # plt.figure(figsize=(8, 6))

    fig = plt.gcf()
    plt.imshow(wordcloud_best, interpolation="bilinear")
    plt.axis("off")
    # plt.show()

    fig.savefig(os.path.join(save_path, f'wc_{model_type}_{split_set}_best.png'), dpi=1000)

    # ST > FT
    wordcloud_worse = WordCloud(max_font_size=50, max_words=len(rs_dict_st), background_color="white",
                            colormap='Oranges',
                        width=1280, height=800).generate_from_frequencies(rs_dict_st)

    # plt.figure(figsize=(8, 6))

    fig = plt.gcf()
    plt.imshow(wordcloud_worse, interpolation="bilinear")
    plt.axis("off")
    # plt.show()

    fig.savefig(os.path.join(save_path, f'wc_{model_type}_{split_set}_worst.png'), dpi=1000)
    
    
    
def merge_column_pairs(ft_report, st_report, ds_name):
    df_pairs_ft = ft_report.get_details(property_name="Column Pair Trends").copy()
    df_pairs_st = st_report.get_details(property_name="Column Pair Trends").copy()

    df_pairs_ft.rename(columns={'Score': 'ft_score'}, inplace=True)
    
    drop_columns = ['Metric', 'Real Correlation', 'Synthetic Correlation']
    if 'Error' in df_pairs_ft.columns:
        ft_drop_columns = drop_columns
        ft_drop_columns.append('Error')
        df_pairs_ft.drop(columns=ft_drop_columns, inplace=True)

    df_pairs_st.rename(columns={'Score': 'st_score'}, inplace=True)
    
    if 'Error' in df_pairs_st.columns:
        st_drop_columns = drop_columns
        st_drop_columns.append('Error')
        df_pairs_st.drop(columns=st_drop_columns, inplace=True)

    df_pairs = df_pairs_ft.merge(df_pairs_st, on=['Column 1', 'Column 2'])

    df_pairs['dataset'] = ds_name
    
    df_pairs = df_pairs[['Column 1', 'Column 2', 'ft_score', 'st_score', 'dataset']]
    df_pairs.dropna(subset=['ft_score', 'st_score'], inplace=True)
    
    return df_pairs

def visualize_colpair(model_type, split_set, df_pairs, save_path, top_k=30):
    
    df_pairs.replace('\n',' ', regex=True, inplace=True)

    df_pairs['ft_st'] = df_pairs['ft_score'] - df_pairs['st_score']
    df_pairs['st_ft'] = df_pairs['st_score'] - df_pairs['ft_score']

    # Remove columns with <string><number> or only <number>
    rules = r'Atr([0-9]*)(.*)|^([\s\d\d+\.\d+]+)$|Unnamed'
    filter = df_pairs['Column 1'].str.contains(rules)
    df_pairs = df_pairs[~filter]
    filter = df_pairs['Column 2'].str.contains(rules)
    df_pairs = df_pairs[~filter]
    
    df_pairs.to_csv(os.path.join(save_path, f'colpair_{model_type}_{split_set}.csv'))

    # Best pairs (FT > ST)
    data_best_pairs = df_pairs.sort_values(by='ft_st', ascending=False)[['Column 1', 'Column 2', 'ft_st']].iloc[:top_k].to_numpy()
    fromto_table_df = pd.DataFrame(data_best_pairs, columns=["from", "to", "value"])
    matrix_best_pairs = Matrix.parse_fromto_table(fromto_table_df)
 
    circos_best = Circos.initialize_from_matrix(
        matrix_best_pairs,
        space=3,
        cmap="viridis",
        # ticks_interval=5,
        label_kws=dict(size=8, r=110, orientation='vertical'),
        link_kws=dict(direction=1, ec="black", lw=0.5),
    )

    fig_best = circos_best.plotfig()
    fig_best.savefig(os.path.join(save_path, f'chord_{model_type}_{split_set}_best.png'), dpi=1000)
    
    # Worst pairs (FT > ST)
    data_worst_pairs = df_pairs.sort_values(by='st_ft', ascending=False)[['Column 1', 'Column 2', 'st_ft']].iloc[:top_k].to_numpy()
    fromto_table_df = pd.DataFrame(data_worst_pairs, columns=["from", "to", "value"])
    matrix_worst_pairs = Matrix.parse_fromto_table(fromto_table_df)
 
    circos_worst = Circos.initialize_from_matrix(
        matrix_worst_pairs,
        space=3,
        cmap="inferno",
        # ticks_interval=5,
        label_kws=dict(size=8, r=110, orientation='vertical'),
        link_kws=dict(direction=1, ec="black", lw=0.5),
    )

    fig_worst = circos_worst.plotfig()
    fig_worst.savefig(os.path.join(save_path, f'chord_{model_type}_{split_set}_worst.png'), dpi=1000)