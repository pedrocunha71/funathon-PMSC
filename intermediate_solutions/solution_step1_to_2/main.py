
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
(
    p9.ggplot(
        data.select(["ccodep", "x", "y", "valeurfonc"]).with_columns(
                valeurfonc_log=pl.col("valeurfonc").log(base=10)
            ).filter(pl.col("ccodep")=="75"),
        p9.aes("x","y", colour="valeurfonc_log")
    ) +
    p9.geom_point(size=0.05)+
    p9.theme_matplotlib() +
    p9.ggtitle("Localization of flat transactions in Paris with price")
)

# %%
(
    p9.ggplot(
        data.select(["x", "y"]),
        p9.aes("x","y")
    ) +
    p9.geom_point(size=0.01)+
    p9.theme_matplotlib() +
    p9.ggtitle("Localization of flat transactions in France since 2010")
)
# %%

# data.filter(
#             pl.col("x") >= -1.599, 
#             pl.col("x") <= -1.598, 
#             pl.col("y") >= 48.838, 
#             pl.col("y") <= 48.839, 
#             pl.col("valeurfonc") <=230000, 
#             pl.col("valeurfonc") >=175000
#             ).sort("anneemut")

# %%

data_h = pl.read_parquet("s3://confpns/synthetic-transactions/rawdata/transactions/transactions_houses_final.parquet")

# %%
# Retrouver la mutation 
# data_h.filter(
#             pl.col("idmutation")=="DVF+_6242255"
#             ).glimpse()


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
            median = serie.median()
            val_min = serie.min()
            val_max = serie.max()
        elif serie.dtype == pl.Date:
            median = serie.median()
            val_min = serie.min()
            val_max = serie.max()
            # Calcul de la date moyenne
            dates = serie.drop_nulls().to_list()
            if dates:
                avg_date = sum((d - datetime.min.date()).days for d in dates) / len(dates)
                median = datetime.min.date() + timedelta(days=avg_date)
            else:
                median = None
        else:  # Strings, booléens, etc.
            median = serie.mode().first()
            val_min = serie.min()
            val_max = serie.max()

        # Conversion de la médiane/mode en string
        # mediane_str = str(mediane) if mediane is not None else "None"

        resultats.append({
            "colonne": col,
            "type": str(serie.dtype),
            "total": n_total,
            "nulls": n_null,
            "NaN": n_nan,
            "valid": n_valid,
            "median/mode": str(median),
            "min": str(val_min),
            "max": str(val_max)
        })

    return pl.DataFrame(resultats)


#%%
descr_df = analyse_colonnes(data)
#%%
print(descr_df)


# %%
# print(descr_df.to_pandas().to_markdown(index=False))


# %%
# Variable to keep
col_to_keep = (
    pl.from_dicts(
        [{"idmutation":0,"datemut":0,"anneemut":0,"moismut":0,"idnatmut":0,
        "libnatmut":0,"valeurfonc":0,"dteloc":0,"jannath":0,"ccodep":0,
        "depcom":0,"x":0,"y":0,"distance_ltm":0,"distance_ltm_corr":0,
        "dnbniv":1,"dnbbai":1,"dnbdou":1,"dnblav":1,"dnbwc":1,"dnbppr":1,
        "dnbsam":1,"dnbcha":1,"dnbcu8":1,"dnbcu9":1,"dnbsea":1,"dnbann":1,
        "dnbpdc":1,"dsupdc":1,"geaulc":0,"gelelc":0,"gesclc":0,"ggazlc":0,
        "gasclc":0,"gchclc":0,"gvorlc":0,"gteglc":0,"dniv":1,"dcntsol":1,
        "dcntagri":1,"dcntnat":1,"nb_garages":1,"nb_piscines":1,
        "nb_terrasses":1,"nb_greniers":1,"nb_caves":1,"nb_autresdep":1}]
    )
    .unpivot()
    .filter(pl.col("value") == 1)
    .select("variable")
    .to_series()
    .to_list()
)
col_to_keep
# %%
# label : stat-des_details_flat
(
    p9.ggplot(
        data
        .select(col_to_keep)
        .unpivot()
        .group_by("variable", "value")
        .agg((pl.len()/1000000).alias("count"))
        .filter(pl.col("value")<10)
        .cast({"value":pl.String}),
        p9.aes(y="count", x="variable", fill="value")
    ) +
    p9.geom_col(position=p9.position_stack(reverse=True)) +
    p9.theme_minimal() +
    p9.coord_flip()
)

# %%
# label : stat-des_details_surf_flat

(
    p9.ggplot(
        data
        .select([
            "dcntsol",
            "dcntnat",
            "dcntagri"
        ])
        .unpivot()
        .filter(pl.col("value")>0),
        p9.aes("value")
    ) +
    p9.geom_histogram(bins = 25, fill='skyblue', color='black')  +
    p9.facet_wrap('~variable', scales='free') +
    p9.theme_minimal()
)


# %%
# Plot price (log) - seems ok 
(
    p9.ggplot(data.with_columns(
                valeurfonc_log=pl.col("valeurfonc").log(base=10)
            ).select(["valeurfonc_log"]).unpivot(value_name="log_price"), 
            p9.aes(x='log_price')
    ) +
    p9.geom_histogram(bins = 100, fill='skyblue', color='black') +
    p9.facet_grid('~variable', scales='free') +
    p9.theme_minimal()
)
# %%
