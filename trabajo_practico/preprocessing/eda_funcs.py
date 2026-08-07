import pandas as pd


def get_frecuency(dataframe, columna):
    freq = dataframe[columna].value_counts()
    freq_rel = dataframe[columna].value_counts(normalize=True).mul(100).round(2)

    freq_tabla = pd.DataFrame(
        {"Frecuencia absoluta": freq, "Frecuencia relativa (%)": freq_rel}
    )

    print(f"Frecuencias de cada categoría en la variable '{columna}':\n{freq_tabla}\n")

    return freq, freq_rel


def get_null_groups(df, group_col, target_col):
    result = (
        df.groupby(group_col, observed=True)[target_col]
        .agg(cantidad_total="size", cantidad_nulos=lambda x: x.isna().sum())
        .reset_index()
    )

    result["porcentaje_nulos"] = (
        result["cantidad_nulos"] / result["cantidad_total"] * 100
    )

    return result.sort_values("porcentaje_nulos", ascending=False)
