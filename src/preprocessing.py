def glossary_feature_selection(path):
    ignore = ['To ignore', 'to ignore',  'Technical characteristic to ignore', 'Aircraft Type (with another regulation -not to be used for the case)']
    columns_df = pd.read_excel(path, sheet_name = 1).iloc[:,[0,2]]
    feature_list = [columns_df.iloc[:,0].tolist()[i] for i in range((len(columns_df.iloc[:,1].tolist()))) if columns_df.iloc[:,1].tolist()[i] not in ignore ]
    return feature_list