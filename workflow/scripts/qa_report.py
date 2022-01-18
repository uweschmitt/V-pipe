#!/usr/bin/env python
# coding: utf-8


import re
import warnings

import numpy as np
import pandas as pd

import sys

input_qafile = sys.argv[1]
output_qafile = sys.argv[2]
coveragefile = sys.argv[3]

warnings.filterwarnings("ignore")

def extract_sample_conditions(samples):
    """Function to parse a list of sample names and return their condition.
    Parameters:
        samples: list of sample names
    Returns:
        (condition, condition_extended): tuple of np.arrays.
        condition is a np.array where 0 = negative control,
        1 = positive control, 2 = experimental condition, -1 = unassigned.
        condition_extended is a np.array where
        0 = H2O negative control, 1 = empty well negative control, 2 = negative PCR tests,
        3 = negative controls of Functional Genomics ZH,
        4 = Twist positive control, 5 = Positive Controls (TO CONFIRM),
        6 = ETHZ_ID sample, 7 = wastewater sample, 8 = functional genomics center sample,
        9= Basel Sequencing of UZH Virology lab, 10 = Labormedizinisches Zentrum Dr Risch, 11 = USZ Tier sample,
        -1 = unassigned
    """
    # make a list of regexps
    re_list = [
        # regexp for H2O negative controls eg: H2O_CP002_A7
        re.compile("^H2O"),
        # regexp for Empty wells negative controls eg: EMPTY_CP002_A11
        re.compile("^((EMPTY)|(empty))"),
        # regexp for samples of negative tests eg: neg_109_B2
        re.compile("^neg_"),
        # regexp for negative controls of Functional Genomics ZH eg: NTC_NA_NTC_NA
        re.compile("^NTC_NA_NTC_NA"),
        # regexp for samples of Twist positive controls eg: pos_MN908947_3_1_100000_CP0002
        re.compile("^(pos_)|(Twist_control)"),
        # regexp for samples of Positive Controls (TO CONFIRM) eg: CoV_ctrl_1_1_10000
        re.compile("CoV_ctrl_"),
        # regexp for ETH ID samples eg: 160000_434_D02
        re.compile("^[0-9]{6}(_Plate){0,1}_(p){0,1}[0-9]+_"),
        # regexp for wastewater samples eg: 09_2020_03_24_B
        re.compile("^[0-9]{2}_202[0-9]_"),
        # regexp for functional genomics eg: 30430668_Plate_8_041120tb3_D7
        re.compile("^[0-9]{8}_Plate_[0-9]+"),
        # regexp for Basel sequencing for some UZH lab (UZH Virology ?) eg: A2_722
        re.compile("^[A-Z][0-9]_[0-9]+"),
        # regexp for lone sample by the lab "Labormedizinisches Zentrum Dr Risch": 674597001
        re.compile("^674597001"),
        # regexp for USZ Tier eg: USZ_5_Tier
        re.compile("^USZ_[0-9]_Tier"),
    ]

    # make 0-1 array with n_samples lines and p_conditions columns
    re_matches = [
        [int(regex.match(sam) is not None) for sam in samples] for regex in re_list
    ]
    re_matches_array = np.array(re_matches)
    print("matches", re_matches)

    # check that samples are assigned to 1 condition max
    if np.max(np.sum(re_matches_array, axis=0)) > 1:
        print("Warning: some samples were assigned to multiple conditions.")

    # assign to each its extended condition
    condition_extended = np.argmax(re_matches_array, axis=0)
    condition_extended = condition_extended - (
        np.sum(re_matches_array, axis=0) == 0
    )  # set unassigned to -1

    # assign to each its condition
    condition = np.in1d(condition_extended, [0, 1, 2, 3]) * 0
    condition = condition + np.in1d(condition_extended, [4, 5]) * 1
    condition = condition + np.in1d(condition_extended, [6, 7, 8, 9, 10, 11]) * 2
    condition = condition + np.in1d(condition_extended, [-1]) * (-1)

    print(condition)
    print(condition_extended)
    return (condition, condition_extended)


# In[ ]:


data = pd.read_csv(input_qafile)
data = data.set_index("sample")
data = data.reset_index().rename(columns={"index": "sample"})

print(data)

coverage = pd.read_csv(
    coveragefile, sep="\t", index_col=["ref", "pos"], compression="gzip"
).droplevel("ref", axis=0)
coverage = coverage.T


data["input_reads"] = data["input_r1"] + data["input_r2"]
data["goodreads"] = data["goodpairs"] * 2
data["conditions"], data["conditions_extended"] = extract_sample_conditions(
    data["sample"]
)
data["group"] = "Real"
data.loc[data.conditions_extended == 0] = "H2O"
data.loc[data.conditions_extended == 1] = "Empty"
data.loc[data.conditions_extended == 2] = "Negative"
data.loc[data.conditions == 2] = "Positive"

print(data)

real_samples = data.loc[data.conditions == 2]
h2o_samples = data.loc[data.conditions_extended == 0]
emp_samples = data.loc[data.conditions_extended == 1]
neg_samples = data.loc[data.conditions_extended == 2]
pos_samples = data.loc[data.conditions == 2]

subsets_dict = dict(
    All=data,
    Real=real_samples,
    Empty=emp_samples,
    H2O=h2o_samples,
    Negative=neg_samples,
    Positive=pos_samples,
)


# In[ ]:


sizes = dict()
for key in subsets_dict:
    sizes[key] = subsets_dict[key].shape[0]
sizes = pd.DataFrame(sizes, index=range(1))

groupsize = sizes.T.reset_index().rename(columns={"index": "group", 0: "size"})


# In[ ]:


coverage_quartiles = coverage.T.describe().loc[["25%", "50%", "75%"]]

coverage_quartiles = coverage_quartiles.T.reset_index().rename(
    columns={"index": "sample"}
)

coverage_quartiles["sample"] = coverage_quartiles["sample"].apply(
    lambda s: s.split("-")[0]
)


# In[ ]:


coverage = coverage.reset_index(drop=True)
real_mean = coverage.loc[data["group"] == "Real"].quantile(q=0.5, axis=0)
real_1q = coverage.loc[data["group"] == "Real"].quantile(q=0.25, axis=0)
real_3q = coverage.loc[data["group"] == "Real"].quantile(q=0.75, axis=0)
empty_mean = coverage.loc[data["group"] == "Empty"].quantile(q=0.5, axis=0)
empty_1q = coverage.loc[data["group"] == "Empty"].quantile(q=0.25, axis=0)
empty_3q = coverage.loc[data["group"] == "Empty"].quantile(q=0.75, axis=0)
h2o_mean = coverage.loc[data["group"] == "H2O"].quantile(q=0.5, axis=0)
h2o_1q = coverage.loc[data["group"] == "H2O"].quantile(q=0.25, axis=0)
h2o_3q = coverage.loc[data["group"] == "H2O"].quantile(q=0.75, axis=0)
neg_mean = coverage.loc[data["group"] == "Negative"].quantile(q=0.5, axis=0)
pos_mean = coverage.loc[data["group"] == "Positive"].quantile(q=0.5, axis=0)


# In[ ]:


coverage_pos_percentiles = pd.DataFrame(columns={"real"})


# In[ ]:


quantity = "basequal"
read_no = 1
quantity = f"r{read_no}_{quantity}"

percs = dict()
for key in subsets_dict:
    percs[key] = subsets_dict[key][f"fastqc_{quantity}"].value_counts(normalize=True)

fqcpercs = pd.DataFrame(percs).fillna(0) * 100
# fqcpercs = fqcpercs.loc[['PASS','WARNING','FAIL','']]


# In[ ]:


upper_bound_reals = 25
lower_bound_controls = 90

percs = dict()
nums = dict()
for key in subsets_dict:
    print(subsets_dict[key]["goodpairs"])
    percs[key] = (
        1.0
        - subsets_dict[key]["goodpairs"]
        / (subsets_dict[key][["input_r1", "input_r2"]].sum(axis=1) / 2)
    ) * 100
    nums[key] = (subsets_dict[key][["input_r1", "input_r2"]].sum() / 2) - subsets_dict[
        key
    ]["goodpairs"]

percs = pd.DataFrame(percs)
nums = pd.DataFrame(nums)

rejreads = percs
rejreads["sample"] = data["sample"]
rejreads["Absolute"] = (
    (data[["input_r1", "input_r2"]].sum(axis=1) / 2) - data["goodpairs"]
).astype(int)


# In[ ]:


lower_bound_reals = 90
upper_bound_controls = 10

percs = dict()
for key in subsets_dict:
    percs[key] = (
        subsets_dict[key]["alnreads"] / (subsets_dict[key]["goodpairs"] * 2.0) * 100
    )

percs = pd.DataFrame(percs)

alnreads = percs
alnreads["sample"] = data["sample"]
alnreads["Absolute"] = data["alnreads"]


# In[ ]:


bwa_insert_dist = data[
    ["sample", "bwa_insert_min", "bwa_insert_mean", "bwa_insert_max"]
]
bwa_insert_dist = bwa_insert_dist.rename(
    columns={
        "bwa_insert_min": "min",
        "bwa_insert_mean": "mean",
        "bwa_insert_max": "max",
    }
)


# In[ ]:


vals_N = dict()
vals_lc = dict()
for key in subsets_dict:
    vals_N[key] = subsets_dict[key]["consensus_N"]
    vals_lc[key] = subsets_dict[key]["consensus_lower"]

vals_N = pd.DataFrame(vals_N)
vals_lc = pd.DataFrame(vals_lc)

lcbases = vals_lc
lcbases["sample"] = data["sample"]

nbases = vals_N
nbases["sample"] = data["sample"]


# In[ ]:


whole_genome_length = 29903
vals = dict()
for key in subsets_dict:
    vals[key] = (
        whole_genome_length
        - subsets_dict[key]["consensus_N"]
        - subsets_dict[key]["match_id"]
    )

vals = pd.DataFrame(vals)

diffbases = vals
diffbases["sample"] = data["sample"]


# In[ ]:


vals_snv = dict()
vals_f = dict()
vals_maj = dict()
for key in subsets_dict:
    vals_snv[key] = subsets_dict[key]["shorah_snv"]
    vals_f[key] = subsets_dict[key]["shorah_filtered"]
    vals_maj[key] = subsets_dict[key]["shorah_majority"]

vals_snv = pd.DataFrame(vals_snv)
vals_f = pd.DataFrame(vals_f)
vals_maj = pd.DataFrame(vals_maj)

filtered_snvs = vals_f
filtered_snvs["sample"] = data["sample"]
maj_snvs = vals_maj
maj_snvs["sample"] = data["sample"]


# In[ ]:


flags = pd.DataFrame({"sample": data["sample"]})
flags_values = pd.DataFrame({"sample": data["sample"]})

flags["coverage"] = "PASS"
assert (coverage_quartiles["sample"] == flags["sample"]).all()
coverage_3q_min_warning = 2000
coverage_3q_min_fail = 1000
coverage_failed = coverage_quartiles.loc[
    coverage_quartiles["75%"] < coverage_3q_min_warning
]["sample"].index
flags.loc[coverage_failed, "coverage"] = "WARNING"
coverage_failed = coverage_quartiles.loc[
    coverage_quartiles["75%"] < coverage_3q_min_fail
]["sample"].index
flags.loc[coverage_failed, "coverage"] = "FAIL"
flags_values["coverage"] = coverage_quartiles["75%"]

flags["r1_basequal"] = data["fastqc_r1_basequal"]
flags["r2_basequal"] = data["fastqc_r2_basequal"]
flags_values["r1_basequal"] = flags["r1_basequal"]
flags_values["r2_basequal"] = flags["r2_basequal"]

flags["rejreads"] = "PASS"
rejupper_bound_warning = 20
rejupper_bound_fail = 40  # or minimum of negative controls?
rejreads_failed = rejreads.loc[rejreads["All"] > rejupper_bound_warning].index
flags.loc[rejreads_failed, "rejreads"] = "WARNING"
rejreads_failed = rejreads.loc[rejreads["All"] > rejupper_bound_fail].index
flags.loc[rejreads_failed, "rejreads"] = "FAIL"
flags_values["rejreads"] = rejreads["All"]

flags["alnreads"] = "PASS"
alnlower_bound = 90
alnreads_failed = alnreads.loc[alnreads["All"] < alnlower_bound].index
flags.loc[alnreads_failed, "alnreads"] = "FAIL"
flags_values["alnreads"] = alnreads["All"]

flags["insertsize"] = "PASS"
insupper_bound = 500
inslower_bound = 300
insertsize_failed = bwa_insert_dist.loc[
    (bwa_insert_dist["min"] < inslower_bound)
    | (bwa_insert_dist["max"] > insupper_bound)
].index
flags.loc[insertsize_failed, "insertsize"] = "FAIL"
flags_values["insertsize"] = bwa_insert_dist["mean"]

flags["consensus_N"] = "PASS"
undupper_bound = 10000
nbases_failed = nbases.loc[nbases["All"] > undupper_bound].index
flags.loc[nbases_failed, "consensus_N"] = "FAIL"
flags_values["consensus_N"] = nbases["All"]

flags["consensus_lcbases"] = "PASS"
ambupper_bound = 20000
lcbases_failed = lcbases.loc[lcbases["All"] > ambupper_bound].index
flags.loc[lcbases_failed, "consensus_lcbases"] = "FAIL"
flags_values["consensus_lcbases"] = lcbases["All"]

flags["consensus_diffbases"] = "PASS"
diffupper_bound_warning = 50
diffupper_bound_fail = 100
diffbases_warning = diffbases.loc[diffbases["All"] > diffupper_bound_warning].index
flags.loc[diffbases_warning, "consensus_diffbases"] = "WARNING"
diffbases_failed = diffbases.loc[diffbases["All"] > diffupper_bound_fail].index
flags.loc[diffbases_failed, "consensus_diffbases"] = "FAIL"
flags_values["consensus_diffbases"] = diffbases["All"]

flags = flags.rename(columns={"snvs_filtered": "snvs"})
flags["snvs"] = "PASS"
snvsupper_bound = 1000
filtered_failed = filtered_snvs.loc[filtered_snvs["All"] > snvsupper_bound].index
flags.loc[filtered_failed, "snvs"] = "FAIL"
flags_values["snvs"] = filtered_snvs["All"]

flags["snvs_majority"] = "PASS"
majupper_bounds = (
    np.max([np.ones(diffbases["All"].shape[0]), diffbases["All"].values], axis=0) * 10
)
majlower_bounds = (
    np.max([np.ones(diffbases["All"].shape[0]), diffbases["All"].values], axis=0) * 0.1
).astype(int)
maj_failed = maj_snvs.loc[
    (maj_snvs["All"].values > majupper_bounds)
    | (maj_snvs["All"].values < majlower_bounds)
].index
flags.loc[maj_failed, "snvs_majority"] = "FAIL"
flags_values["snvs_majority"] = maj_snvs["All"]

flags = flags.rename(columns={"sample": "Sample"})
flags_values = flags_values.rename(columns={"sample": "Sample"})


# In[ ]:


splt = data["batch"].str.split("_", n=1, expand=True)
data["date"] = splt[0]
data["flowcell"] = splt[1]
date = str(pd.to_datetime(data.loc[0, "date"], format="%Y%m%d").date())


# ## Summary

# In[ ]:


# List of real samples that failed for consensus usage
# Sample is failed if any of the flags is a fail
sample_status = pd.DataFrame(dict(samples=flags["Sample"], status="PASS"))
for index, row in flags.iterrows():
    if np.any(np.array(row.values[:-2]) == "FAIL"):
        sample_status.loc[index, "status"] = "FAIL"
    elif np.any(np.array(row.values[:-2]) == "WARNING"):
        sample_status.loc[index, "status"] = "WARNING"

failed_reals = sample_status.loc[data["group"] == "Real"].loc[
    sample_status["status"] == "FAIL"
]["samples"]

print(f"\tFAILED {len(failed_reals)} real samples:\n")
for sample in failed_reals:
    print("\t" + sample)


sys.exit(0)

import plotly.figure_factory as ff
import plotly.graph_objs as go
import plotly.io as pio
import seaborn as sns

sns.set_style("ticks")


group_colors = dict(
    Real="#80b1d3",
    Empty="#ffffb3",
    H2O="#bebada",
    Negative="#fb8072",
    Positive="#8dd3c7",
)

layout = go.Layout(
    title="Sample group sizes",
    xaxis=go.layout.XAxis(title="Group", showticklabels=True),
    yaxis=go.layout.YAxis(title="Number of samples"),
    margin=go.layout.Margin(t=50),
    template="plotly_white",
)

colors = ["#3182bd"] + list(group_colors.values())
fig = go.Figure()

# Add traces
fig.add_trace(
    go.Bar(x=groupsize["group"], y=groupsize["size"], marker=dict(color=colors))
)

fig.update_layout(layout)
fig.show()


# In[ ]:


colorscale = [
    [0, "red"],
    [0.111111111111, "yellow"],
    [0.222222222222, "green"],
    [0.333333333333, "rgb(141,211,199)"],
    [0.444444444444, "rgb(255,255,179)"],
    [0.555555555556, "rgb(190,186,218)"],
    [0.666666666667, "rgb(190,186,218)"],
    [0.777777777778, "rgb(128,177,211)"],
    [0.888888888889, "rgb(128,177,211)"],
    [1, "rgb(49,54,149)"],
]
passcolordicts = {"FAIL": 0.0, "WARNING": 1.0 / 9.0, "PASS": 2.0 / 9.0}
groupcolordicts = {
    "Real": 7.0 / 9.0,
    "Empty": 6.0 / 9.0,
    "H2O": 5.0 / 9.0,
    "Negative": 4.0 / 9.0,
    "Positive": 3.0 / 9.0,
}
colors = np.array(
    [[passcolordicts[v] for v in flags[s].values] for s in list(flags.columns[1:])]
)
colors = np.concatenate(
    [np.array([[groupcolordicts[group] for group in data["group"]]]), colors]
)
colors = np.concatenate(
    [np.array([[8.0 / 9.0 for s in list(flags.columns)]]), colors.T]
)


# In[ ]:


flags2 = flags.rename(
    columns={
        "coverage": "Coverage",
        "r1_basequal": "Base quality R1",
        "r2_basequal": "Base quality R2",
        "rejreads": "Rejected reads",
        "alnreads": "Aligned reads",
        "insertsize": "Insert size",
        "consensus_N": "Consensus<br>undetermined",
        "consensus_lcbases": "Consensus<br>ambiguousity",
        "consensus_diffbases": "Consensus<br>sequence",
        "snvs": "Number of<br>SNVs",
        "snvs_majority": "Majority<br>SNVs",
    }
)
from copy import deepcopy

flags3 = deepcopy(flags2)
flags3.loc[np.where(np.array(["pos" in f for f in flags3["Sample"]]))[0], "Sample"] = [
    "_".join(a.split("_", 3)[:2])
    for a in flags3.loc[
        np.where(np.array(["pos" in f for f in flags3["Sample"]]))[0], "Sample"
    ].tolist()
]
flags3.loc[
    np.where(np.array(["EMPTY" in f for f in flags3["Sample"]]))[0], "Sample"
] = [
    "_".join(a.split("_", 3)[:2])
    for a in flags3.loc[
        np.where(np.array(["EMPTY" in f for f in flags3["Sample"]]))[0], "Sample"
    ].tolist()
]


# In[ ]:


descriptions = {
    "Sample": "",
    "Coverage": f"Third quartile of the coverage across the whole genome.<br>WARNING if < {coverage_3q_min_warning}<br>FAIL if < {coverage_3q_min_fail}",
    "Base quality R1": "FastQC per base quality flag for read 1",
    "Base quality R2": "FastQC per base quality flag for read 2",
    "Rejected reads": f"Percentage of reads rejected by Prinseq.<br>WARNING if > {rejupper_bound_warning}%<br>FAIL if > {rejupper_bound_fail}%",
    "Aligned reads": f"Percentage of kept reads that were aligned.<br>FAIL if < {alnlower_bound}%",
    "Insert size": f"Minimum and maximum estimated insert sizes.<br>FAIL if minimum < {inslower_bound} or maximum > {insupper_bound}",
    "Consensus<br>undetermined": f"Number of bases considered undetermined, i.e., with support in less than 5 reads.<br>FAIL if > {undupper_bound}",
    "Consensus<br>ambiguosity": f"Number of bases considered ambiguous, i.e., with support in less than 50 reads.<br>FAIL if > {ambupper_bound}",
    "Consensus<br>sequence": f'Number of bases different from the reference sequence. "N" calls not included.<br>WARNING if > {diffupper_bound_warning}<br>FAIL if > {diffupper_bound_fail}',
    "Number of<br>SNVs": f"Number of single-nucleotide variants<br>FAIL if > {snvsupper_bound}",
    "Majority<br>SNVs": f"Number of consensus single-nucleotide variants<br>FAIL if 10x larger than the consensus sequence",
}

hover = []
hover.append(["<b>" + desc + "</b>" for desc in list(descriptions.values())])
for i in range(flags_values.shape[0]):
    row = [flags_values.iloc[i]["Sample"]]
    for j in range(1, flags_values.shape[1]):
        val = str(flags_values.iloc[i, j])
        if (
            flags_values.columns[j] == "rejreads"
            or flags_values.columns[j] == "alnreads"
        ):
            val = f"{flags_values.iloc[i, j]:.2f}"
        row.append(
            "<b>"
            + list(descriptions.values())[j]
            + "</b>"
            + "<br><br>"
            + "<b>"
            + str(flags_values.iloc[i, 0])
            + ": </b>"
            + val
            + "<br><br>"
            + "Coverage 3Q: "
            + str(flags_values.iloc[i]["coverage"])
            + "<br>"
            + f"% Rejected reads: {flags_values.iloc[i]['rejreads']:.2f} <br>"
            + f"% Aligned reads: {flags_values.iloc[i]['alnreads']:.2f}"
        )

    hover.append(row)


# In[ ]:


flags_values.to_csv(output_qafile, index=False)


# In[ ]:


table_simple = ff.create_table(
    flags3,
    text=hover,
    hoverinfo="text",
)
#                             hovertemplate="<b>%{text}</b><br><br>%{y}<br><br>" +
#                         "<extra></extra>")
table_simple.update_layout(
    autosize=False,
    width=1400,
)
d = table_simple.to_dict()
d["data"][0]["z"] = colors
d["data"][0]["colorscale"] = colorscale

pio.show(d, config={"displayModeBar": False})


# ## Coverage

# In[ ]:


delta_qs = coverage_quartiles.T.iloc[1:].T.diff(axis=1)
delta_qs["sample"] = coverage_quartiles["sample"]
delta_qs["25%"] = coverage_quartiles["25%"]


# In[ ]:


layout = go.Layout(
    title="Coverage quartiles per sample",
    xaxis=go.layout.XAxis(title="Samples", showticklabels=False),
    yaxis=go.layout.YAxis(title="Coverage"),
    template="plotly_white",
    barmode="relative",
)

colors = ["#6baed6", "#3182bd", "#08519c"]
fig = go.Figure()

# Add traces
fig.add_trace(
    go.Bar(
        x=coverage_quartiles["sample"],
        y=delta_qs["25%"],
        # mode='lines',
        text=coverage_quartiles["25%"],
        hovertemplate="(%{x}, %{text})<extra></extra>",
        name="1Q",
        marker=dict(color=colors[0]),
    )
)
fig.add_trace(
    go.Bar(
        x=coverage_quartiles["sample"],
        y=delta_qs["50%"],
        # mode='lines',
        text=coverage_quartiles["50%"],
        hovertemplate="(%{x}, %{text})<extra></extra>",
        name="Median",
        marker=dict(color=colors[1]),
    )
)
fig.add_trace(
    go.Bar(
        x=coverage_quartiles["sample"],
        y=delta_qs["75%"],
        # mode='lines',
        text=coverage_quartiles["75%"],
        hovertemplate="(%{x}, %{text})<extra></extra>",
        name="3Q",
        marker=dict(color=colors[2]),
    )
)
fig.add_trace(
    go.Scatter(
        x=coverage_quartiles["sample"],
        y=np.ones((coverage_quartiles.shape[0])) * coverage_3q_min_warning,
        mode="lines",
        name="Soft lower bound\nfor 3Q",
        marker=dict(color="#CCCC00"),
        line=dict(dash="dot"),
    )
)
fig.add_trace(
    go.Scatter(
        x=coverage_quartiles["sample"],
        y=np.ones((coverage_quartiles.shape[0])) * coverage_3q_min_fail,
        mode="lines",
        name="Hard lower bound\nfor 3Q",
        marker=dict(color="red"),
        line=dict(dash="dot"),
    )
)

fig.update_layout(layout)

fig.show()


# In[ ]:


group_percentiles = dict(
    Real=pd.DataFrame(
        {
            "25%": coverage.loc[data["group"] == "Real"].quantile(q=0.25, axis=0),
            "50%": coverage.loc[data["group"] == "Real"].quantile(q=0.5, axis=0),
            "75%": coverage.loc[data["group"] == "Real"].quantile(q=0.75, axis=0),
        }
    ),
    Empty=pd.DataFrame(
        {
            "25%": coverage.loc[data["group"] == "Empty"].quantile(q=0.25, axis=0),
            "50%": coverage.loc[data["group"] == "Empty"].quantile(q=0.5, axis=0),
            "75%": coverage.loc[data["group"] == "Empty"].quantile(q=0.75, axis=0),
        }
    ),
    H2O=pd.DataFrame(
        {
            "25%": coverage.loc[data["group"] == "H2O"].quantile(q=0.25, axis=0),
            "50%": coverage.loc[data["group"] == "H2O"].quantile(q=0.5, axis=0),
            "75%": coverage.loc[data["group"] == "H2O"].quantile(q=0.75, axis=0),
        }
    ),
    Negative=pd.DataFrame(
        {
            "25%": coverage.loc[data["group"] == "Negative"].quantile(q=0.25, axis=0),
            "50%": coverage.loc[data["group"] == "Negative"].quantile(q=0.5, axis=0),
            "75%": coverage.loc[data["group"] == "Negative"].quantile(q=0.75, axis=0),
        }
    ),
    Positive=pd.DataFrame(
        {
            "25%": coverage.loc[data["group"] == "Positive"].quantile(q=0.25, axis=0),
            "50%": coverage.loc[data["group"] == "Positive"].quantile(q=0.5, axis=0),
            "75%": coverage.loc[data["group"] == "Positive"].quantile(q=0.75, axis=0),
        }
    ),
)


# In[ ]:


colors = ["#6baed6", "#3182bd", "#08519c"]

groups = data["group"].unique()
vals = []
for i, g in enumerate(groups):
    a = [False, False, False] * len(groups)
    a[i * 3 : i * 3 + 3] = [True, True, True]
    vals.append(a)

# Add traces
fig = go.Figure()
for i, group in enumerate(groups):
    for j, p in enumerate(["25%", "50%", "75%"]):
        fig.add_trace(
            go.Scatter(
                x=group_percentiles[group].index,
                y=group_percentiles[group][p],
                mode="lines",
                name=["1st quart.", "Median", "3rd quart."][j],
                marker=dict(color=colors[j]),
                visible=False if i > 0 else True,
            )
        )

updatemenus = list(
    [
        dict(
            active=0,
            showactive=True,
            buttons=list(
                [
                    dict(label=group, method="update", args=[{"visible": vals[i]}])
                    for i, group in enumerate(groups)
                ]
            ),
            direction="down",
            pad={"r": 10, "t": 10},
            x=0.05,
            xanchor="left",
            y=1.12,
            yanchor="top",
        )
    ]
)

layout = go.Layout(
    title="Coverage quartiles per position per group",
    xaxis=go.layout.XAxis(title="Position", showticklabels=True),
    yaxis=go.layout.YAxis(title="Coverage"),
    template="plotly_white",
    updatemenus=updatemenus,
)

fig.update_layout(layout)
# Add annotation
fig.update_layout(
    annotations=[
        dict(text="Group:", showarrow=False, x=0.0, y=1.085, yref="paper", align="left")
    ]
)

fig.show()


# ## Prinseq

# In[ ]:


layout = go.Layout(
    title="Reads rejected by Prinseq",
    xaxis=go.layout.XAxis(title="Samples", showticklabels=False),
    yaxis=go.layout.YAxis(title="Reads rejected (%)"),
    template="plotly_white",
)

# fig = go.Figure(data=[{'y': coverage_quartiles['coverage'], 'x': coverage_quartiles['sample']}], layout=layout)
# iplot(fig)
colors = ["#3182bd"]
fig = go.Figure()

# Add traces
fig.add_trace(
    go.Bar(
        x=rejreads["sample"],
        y=rejreads["All"],
        name="Observed",
        marker=dict(color=colors[0]),
        text=rejreads["Absolute"].values,
        hoverinfo="y+text",
        hovertemplate="<b>%{x}</b><br><br>"
        + "Percentage: %{y:.2f}%<br>"
        + "Absolute: %{text}<br>"
        "<extra></extra>",
    )
)
fig.add_trace(
    go.Scatter(
        x=rejreads["sample"],
        y=np.ones((rejreads.shape[0])) * rejupper_bound_warning,
        mode="lines",
        name="Soft upper bound",
        marker=dict(color="#CCCC00"),
        line=dict(dash="dot"),
    )
)
fig.add_trace(
    go.Scatter(
        x=rejreads["sample"],
        y=np.ones((rejreads.shape[0])) * rejupper_bound_fail,
        mode="lines",
        name="Hard upper bound",
        marker=dict(color="red"),
        line=dict(dash="dot"),
    )
)
fig.update_layout(layout)
fig.show()
# fig = px.line(coverage_quartiles, x="sample", y="coverage",
#               color='percentile', hover_name="sample", template="plotly_white")

# iplot(fig)


# ## Alignment

# In[ ]:


layout = go.Layout(
    title="Reads aligned by BWA",
    xaxis=go.layout.XAxis(title="Samples", showticklabels=False),
    yaxis=go.layout.YAxis(title="Reads aligned (%)"),
    template="plotly_white",
)

colors = ["#3182bd"]
fig = go.Figure()

# Add traces
fig.add_trace(
    go.Bar(
        x=alnreads["sample"],
        y=alnreads["All"].values,
        name="Observed",
        marker=dict(color=colors[0]),
        text=alnreads["Absolute"].values,
        hoverinfo="y+text",
        hovertemplate="<b>%{x}</b><br><br>"
        + "Percentage: %{y:.2f}%<br>"
        + "Absolute: %{text}<br>"
        "<extra></extra>",
    )
)
fig.add_trace(
    go.Scatter(
        x=alnreads["sample"],
        y=np.ones((alnreads.shape[0])) * 90,
        mode="lines",
        name="Lower bound",
        marker=dict(color="red"),
        line=dict(dash="dot"),
    )
)

fig.update_layout(layout)
fig.show()


# In[ ]:


delta_is = bwa_insert_dist.T.iloc[1:].T.diff(axis=1)
delta_is["sample"] = bwa_insert_dist["sample"]
delta_is["min"] = bwa_insert_dist["min"]


# In[ ]:


layout = go.Layout(
    title="Insert sizes estimated by BWA",
    xaxis=go.layout.XAxis(title="Samples", showticklabels=False),
    yaxis=go.layout.YAxis(title="Insert sizes", range=[0, 600]),
    template="plotly_white",
    barmode="stack",
)

colors = ["#6baed6", "#3182bd", "#08519c"]
fig = go.Figure()

# Add traces
fig.add_trace(
    go.Bar(
        x=bwa_insert_dist["sample"],
        y=delta_is["min"],
        # mode='lines',
        text=bwa_insert_dist["min"],
        hovertemplate="(%{x}, %{text})<extra></extra>",
        name="Min",
        marker=dict(color=colors[0]),
    )
)
fig.add_trace(
    go.Bar(
        x=bwa_insert_dist["sample"],
        y=delta_is["mean"],
        # mode='lines',
        text=bwa_insert_dist["mean"],
        hovertemplate="(%{x}, %{text})<extra></extra>",
        name="Mean",
        marker=dict(color=colors[1]),
    )
)
fig.add_trace(
    go.Bar(
        x=bwa_insert_dist["sample"],
        y=delta_is["max"],
        # mode='lines',
        text=bwa_insert_dist["max"],
        hovertemplate="(%{x}, %{text})<extra></extra>",
        name="Max",
        marker=dict(color=colors[2]),
    )
)

fig.add_trace(
    go.Scatter(
        x=rejreads["sample"],
        y=np.ones((rejreads.shape[0])) * 300,
        mode="lines",
        name="Lower bound",
        marker=dict(color="red"),
        line=dict(dash="dot"),
    )
)
fig.add_trace(
    go.Scatter(
        x=rejreads["sample"],
        y=np.ones((rejreads.shape[0])) * 500,
        mode="lines",
        name="Upper bound",
        marker=dict(color="red"),
        line=dict(dash="dot"),
    )
)

fig.update_layout(layout)
fig.show()


# ## Consensus sequence

# In[ ]:


layout = go.Layout(
    title="Number of undetermined (N) bases in consensus",
    xaxis=go.layout.XAxis(title="Samples", showticklabels=False),
    yaxis=go.layout.YAxis(title="Number of N calls"),
    template="plotly_white",
)

colors = ["#3182bd"]
fig = go.Figure()

# Add traces
fig.add_trace(
    go.Bar(
        x=nbases["sample"],
        y=nbases["All"],
        name="Observed",
        marker=dict(color=colors[0]),
    )
)
fig.add_trace(
    go.Scatter(
        x=nbases["sample"],
        y=np.ones((nbases.shape[0])) * undupper_bound,
        mode="lines",
        name="Upper bound",
        marker=dict(color="red"),
        line=dict(dash="dot"),
    )
)

fig.update_layout(layout)
fig.show()


# In[ ]:


layout = go.Layout(
    title="Number of ambiguous (lowercase) bases in consensus",
    xaxis=go.layout.XAxis(title="Samples", showticklabels=False),
    yaxis=go.layout.YAxis(title="Number of lowercase calls"),
    template="plotly_white",
)

colors = ["#3182bd"]
fig = go.Figure()

# Add traces
fig.add_trace(
    go.Bar(
        x=lcbases["sample"],
        y=lcbases["All"],
        name="Observed",
        marker=dict(color=colors[0]),
    )
)
fig.add_trace(
    go.Scatter(
        x=lcbases["sample"],
        y=np.ones((lcbases.shape[0])) * ambupper_bound,
        mode="lines",
        name="Upper bound",
        marker=dict(color="red"),
        line=dict(dash="dot"),
    )
)

fig.update_layout(layout)
fig.show()


# In[ ]:


layout = go.Layout(
    title='Number of different bases* between consensus and reference<br><span style="font-size:12">*not including "N"</span>',
    xaxis=go.layout.XAxis(title="Samples", showticklabels=False),
    yaxis=go.layout.YAxis(
        title="Number of bases", range=[0, diffupper_bound_fail + 10]
    ),
    template="plotly_white",
)
colors = ["#3182bd"]
fig = go.Figure()

# Add traces
fig.add_trace(
    go.Bar(
        x=diffbases["sample"],
        y=diffbases["All"],
        name="Observed",
        marker=dict(color=colors[0]),
    )
)
fig.add_trace(
    go.Scatter(
        x=diffbases["sample"],
        y=np.ones((diffbases.shape[0])) * diffupper_bound_warning,
        mode="lines",
        name="Soft upper bound",
        marker=dict(color="#CCCC00"),
        line=dict(dash="dot"),
    )
)

fig.add_trace(
    go.Scatter(
        x=diffbases["sample"],
        y=np.ones((diffbases.shape[0])) * diffupper_bound_fail,
        mode="lines",
        name="Hard upper bound",
        marker=dict(color="red"),
        line=dict(dash="dot"),
    )
)
fig.update_layout(layout)
fig.show()


# ## ShoRAH

# In[ ]:


layout = go.Layout(
    title="SNVs called by ShoRAH (posterior probability > 80%)",
    xaxis=go.layout.XAxis(title="Samples", showticklabels=False),
    yaxis=go.layout.YAxis(title="Number of SNVs"),
    template="plotly_white",
)

colors = ["#3182bd"]
fig = go.Figure()

# Add traces
fig.add_trace(
    go.Bar(
        x=filtered_snvs["sample"],
        y=filtered_snvs["All"],
        name="Observed",
        marker=dict(color=colors[0]),
    )
)
fig.add_trace(
    go.Scatter(
        x=filtered_snvs["sample"],
        y=np.ones((filtered_snvs.shape[0])) * 2000,
        mode="lines",
        name="Upper bound",
        marker=dict(color="red"),
        line=dict(dash="dot"),
    )
)

fig.update_layout(layout)
fig.show()


# In[ ]:


layout = go.Layout(
    title="Number of SNVs from filtered ShoRAH-based majority vote",
    xaxis=go.layout.XAxis(title="Samples", showticklabels=False),
    yaxis=go.layout.YAxis(title="Number of SNVs"),
    template="plotly_white",
    yaxis_type="log",
)

colors = ["#3182bd"]
fig = go.Figure()

# Add traces
fig.add_trace(
    go.Bar(
        x=maj_snvs["sample"],
        y=maj_snvs["All"],
        name="Observed",
        marker=dict(color=colors[0]),
    )
)
fig.add_trace(
    go.Scatter(
        x=maj_snvs["sample"],
        y=majupper_bounds,
        mode="markers",
        name="Upper bound",
        marker=dict(color="red"),
        line=dict(dash="dot"),
    )
)

fig.update_layout(layout)
fig.show()
