
# ============================================
# STEP 3 — Train / test split, model fitting,
#          and performance evaluation
# ============================================

# YOUR CODE HERE
# %%
import polars as pl
import plotnine as p9
from datetime import datetime, timedelta

pl.Config.set_tbl_cols(-1)  # show all columns
pl.Config.set_tbl_rows(50)  # show 20 rows
# %%
data = pl.read_parquet('s3://confpns/synthetic-transactions/rawdata/transactions/transactions_flats_final.parquet')
data
# %%
data2 = data.filter(
            pl.col("ccodep")=="75"
            ).select(["x", "y", "valeurfonc"]).with_columns(
                valeurfonc_log=pl.col("valeurfonc").log(base=10)
            )
# %%
(
    p9.ggplot(
        data2,
        p9.aes("x","y", colour="valeurfonc_log")
    ) +
    p9.geom_point(size=0.05)+
    p9.theme_matplotlib()
)
# %%
# Retrouver la mutation 
(
    p9.ggplot(
        data.filter(
            pl.col("x")>=-1.599, 
            pl.col("x") <= -1.598, 
            pl.col("y") >= 48.838, 
            pl.col("y") <= 48.839, 
            pl.col("valeurfonc") <=150000
            ).select(["x", "y", "anneemut", "valeurfonc"]),
        p9.aes("x","y", size="anneemut", colour="valeurfonc")
        ) +
    p9.geom_point() 
)

data.filter(
            pl.col("x")>=-1.599, 
            pl.col("x") <= -1.598, 
            pl.col("y") >= 48.838, 
            pl.col("y") <= 48.839, 
            pl.col("valeurfonc") <=150000, 
            pl.col("valeurfonc") >=100000
            ).sort("anneemut")

        # idmutation : "DVF+_3356048"

# %%

data_h = pl.read_parquet("s3://confpns/synthetic-transactions/rawdata/transactions/transactions_houses_final.parquet")

# %%
# Retrouver la mutation 
data_h.filter(
            pl.col("idmutation")=="DVF+_6242255"
            ).glimpse()
# %%


# %%
def analyse_colonnes(df: pl.DataFrame) -> pl.DataFrame:
    """
    Retourne un DataFrame avec pour chaque colonne :
    - Nom, type, statistiques de valeurs (nulles, NaN, manquantes, valides)
    - Médiane (numérique), mode (string), date moyenne (date)
    - Min, max
    """
    resultats = []

    for col in df.columns:
        serie = df[col]
        n_total = len(serie)
        n_null = serie.null_count()

        # Calcul des NaN (spécifique aux float)
        n_nan = 0
        if serie.dtype in (pl.Float32, pl.Float64):
            n_nan = serie.is_nan().sum()
        n_valid = n_total - n_null - n_nan

        # Calcul de la médiane, mode, min, max, ou date moyenne
        if serie.dtype in (pl.Int8, pl.Int16, pl.Int32, pl.Int64, pl.Float32, pl.Float64):
            mediane = serie.median()
            val_min = serie.min()
            val_max = serie.max()
        elif serie.dtype == pl.Date:
            mediane = serie.median()
            val_min = serie.min()
            val_max = serie.max()
            # Calcul de la date moyenne
            dates = serie.drop_nulls().to_list()
            if dates:
                avg_date = sum((d - datetime.min.date()).days for d in dates) / len(dates)
                mediane = datetime.min.date() + timedelta(days=avg_date)
            else:
                mediane = None
        else:  # Strings, booléens, etc.
            mediane = serie.mode().first()
            val_min = serie.min()
            val_max = serie.max()

        # Conversion de la médiane/mode en string
        # mediane_str = str(mediane) if mediane is not None else "None"

        resultats.append({
            "colonne": col,
            "type": str(serie.dtype),
            "total": n_total,
            "nulles": n_null,
            "NaN": n_nan,
            "valides": n_valid,
            "médiane/mode": str(mediane),
            "min": str(val_min),
            "max": str(val_max)
        })

    return pl.DataFrame(resultats)


#%%
descr_df = analyse_colonnes(data)
#%%
print(descr_df)
# print(descr_df.to_pandas().to_markdown(index=False))

# %%

