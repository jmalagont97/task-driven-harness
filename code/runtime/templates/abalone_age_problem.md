# Task: estimate the number of rings of one abalone specimen of sex {{sex}}

We want to know the number of growth rings (and so the age) of an abalone
(genus Haliotis) whose sex is known.

## Measurements for the tool agents (dataset units)

These are the values exactly as they appear in the dataset. The dataset scaled
every continuous measurement by dividing it by 200, and the tools were trained
on these scaled values. Send only these values to the tools.

{{dataset_values}}

## Measurements for the literature experts (original units)

The same specimen in the units used by the articles: millimetres and grams
(dataset value multiplied by 200). Use only these values when you compare the
specimen with your article.

{{original_values}}

## What the variables mean

{{variables_text}}

## About the data

{{about_text}}

Answer with a number of rings and an 80% range (age in years = rings + 1.5),
or say that the evidence cannot decide.
