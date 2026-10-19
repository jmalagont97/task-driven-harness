# Key Facts for Checking the UCI Abalone Dataset Against Its Source Literature

This file is not part of any source publication. It collects verifiable facts from the four documents in this collection, each with its citation and page, for checking the UCI Abalone dataset. Use it as a separate document in a RAG index or as context in a system prompt.

Sources:
- Nash et al. (1994): The Population Biology of Abalone (Haliotis species) in Tasmania. I. Blacklip Abalone (H. rubra) from the North Coast and the Islands of Bass Strait. Technical Report No. 48.
- Prince et al. (1988a): A method for ageing the abalone Haliotis rubra. Aust. J. Mar. Freshwater Res. 39: 167-175.
- Prince et al. (1988b): Recruitment, growth, mortality and population structure in a southern Australian population of Haliotis rubra. Marine Biology 100: 75-82.
- McShane and Smith (1992): Shell growth checks are unreliable indicators of age of the abalone Haliotis rubra. Aust. J. Mar. Freshwater Res. 43: 1215-1219.

## 1. Dataset columns and the original measurements

| Dataset column | Original measurement (Nash et al. 1994, p. 8) | Notes |
|---|---|---|
| Sex (M, F, I) | Sexual condition: male, female or immature, determined by visual inspection of the intact gonad; uncertain cases resolved under the microscope (p. 10) | I = immature (no discernible testis or ovary). It is a maturity state, not a sex. |
| Length | Maximum shell length (mm) | In the UCI version, continuous values are divided by 200. |
| Diameter | Minimum shell length (mm), perpendicular to length | Divided by 200 in UCI. |
| Height | Vertical shell height (mm), with meat in the shell | Divided by 200 in UCI. |
| Whole_weight | Whole weight (g) | Divided by 200 in UCI. |
| Shucked_weight | Shucked meat weight (g) | Divided by 200 in UCI. |
| Viscera_weight | Viscera weight (g) | Divided by 200 in UCI. |
| Shell_weight | Dry shell weight (g), dried to constant weight at 90 C | Divided by 200 in UCI. |
| Rings | Number of major growth rings in the shell spire | Age (years) = rings + 1.5 (Nash et al. 1994, p. 9), assuming one major ring per year. |

Known gaps between the report and the dataset:
- Nash et al. (1994, Table 3) sampled 4754 abalone (2064 female, 1827 male, 833 immature, 15 parasitised). The UCI dataset has 4177 records (M 1528, F 1307, I 1342) after removal of records with missing values. The sex proportions differ, and none of the four documents explains how the dataset was derived.
- The dataset has no column for survey area, although growth, size at maturity and the length-weight relationship differ markedly between the five areas (Nash et al. 1994, Tables 1, 4 and 5).
- Only 33-48% of shells could be aged in the survey (Nash et al. 1994, Table 3).

## 2. Age and growth rings

- Method: the spire is ground flat to expose a disc of nacre, which is polished (and, in later studies, acid-etched and stained); rings are counted under a dissecting microscope (Prince et al. 1988a, p. 170; Nash et al. 1994, p. 9; McShane and Smith 1992, p. 1216).
- Three minor rings form at about 7.4, 11.4 and 15.7 months; the first major ring at about 20.6 months; later major rings at approximately annual intervals (32.7, 43.4, 59.2, 69.3, 79.8 and 89.3 months) (Prince et al. 1988a, pp. 171-172).
- Major rings form in winter (June-August), not at spawning (Prince et al. 1988a, p. 172).
- The 1.5-year offset accounts for the period in which the minor rings are deposited (Nash et al. 1994, p. 9).
- Minor rings can be lost when the spire of large shells (>90 mm) is ground (Prince et al. 1988a, p. 171).
- Maximum major ring counts observed in Bass Strait: about 23-29 depending on the area (Nash et al. 1994, Fig. 15). This matches the 1-29 range of Rings in the UCI dataset.
- Length varies widely at a given ring count, for example about 60-115 mm at about 10 years at Babel Island (Nash et al. 1994, Fig. 16).

Points of disagreement:
- McShane and Smith (1992) found that growth checks were not annual in three Victorian populations, and that ring counts underestimated growth at one site and overestimated it for large abalone at another (pp. 1216-1218).
- Prince et al. (1988a, pp. 173-174) found the interpretation valid for Tasmanian and Victorian samples but possibly not for New South Wales.
- Nash et al. (1994, pp. 54-55) consider McShane and Smith's general conclusion premature: marginal increment analysis in Tasmania indicates one layer per year. Nash et al. repeated their analyses with 1 and 2 rings per year (p. 9).

## 3. Sexual maturity (relevant to the Infant category)

Length and age at 50% maturity (Nash et al. 1994, Table 5, p. 31):

| Area | a (length) | b (length) | L50 (mm) | 95% CL of L50 | a (age) | b (age) | Age at 50% maturity (years) |
|---|---|---|---|---|---|---|---|
| Hogan Is. | −12.770 | 0.155 | 82.6 | 79.8-85.1 | −6.558 | 0.692 | 9.5 |
| Kent Is. | −8.073 | 0.110 | 73.6 | 70.7-76.0 | −4.750 | 0.482 | 9.9 |
| Babel I. | −8.517 | 0.123 | 69.0 | 65.6-72.0 | −7.082 | 0.880 | 8.0 |
| Cape Barren I. | −10.960 | 0.145 | 75.6 | 73.0-78.0 | −5.707 | 0.628 | 9.1 |
| Waterhouse I. | −11.120 | 0.170 | 65.3 | 61.8-68.3 | −12.290 | 1.379 | 8.9 |

- Proportion mature at length L (mm): p = e^(a + bL) / (1 + e^(a + bL)) (Nash et al. 1994, Equation 2, p. 10). Example for the Hogan Islands: L = 60 mm gives p ≈ 0.03; L = 100 mm gives p ≈ 0.94.
- Gonad development is first seen at 45-50 mm (Waterhouse I.) to 60-65 mm (Hogan Is.); full maturity at about 90 mm (Waterhouse) to about 120 mm (Hogan) (Nash et al. 1994, p. 31).
- Age at 50% maturity is 8.0-9.9 years, i.e. about 6.5-8.5 major rings when age = rings + 1.5; all animals are mature only in year classes 13-15 and older (Nash et al. 1994, p. 31).
- At a given ring count, mature abalone are always larger than immature ones, typically by 10-30 mm (Nash et al. 1994, Fig. 19, p. 30).
- In southern Tasmania (faster growth), maturity begins at about 90 mm and 4-5 years, and most abalone are mature above 110 mm and at 6-7 years (Prince et al. 1988b, pp. 79, 81).

## 4. Length-weight relationships

Whole weight (g) = a × (shell length, mm)^b (Nash et al. 1994, Table 1, p. 19):

| Area | a | b | r² | n |
|---|---|---|---|---|
| Hogan Is. | 0.0001495 | 2.962 | 0.971 | 1097 |
| Kent Is. | 0.0000831 | 3.087 | 0.974 | 1069 |
| Babel I. | 0.0001083 | 3.040 | 0.973 | 1015 |
| Cape Barren I. | 0.0000773 | 3.122 | 0.976 | 867 |
| Waterhouse I. | 0.0000888 | 3.095 | 0.983 | 633 |

Shucked weight (g) = a × (shell length, mm)^b (Nash et al. 1994, Table 2, p. 19):

| Area | a | b | r² | n |
|---|---|---|---|---|
| Hogan Is. | 0.0000698 | 2.928 | 0.936 | 1087 |
| Kent Is. | 0.0000484 | 2.996 | 0.954 | 1064 |
| Babel I. | 0.0000886 | 2.882 | 0.959 | 1011 |
| Cape Barren I. | 0.0000407 | 3.045 | 0.965 | 856 |
| Waterhouse I. | 0.0000287 | 3.104 | 0.965 | 625 |

Example: at the Hogan Islands, a 100 mm abalone is expected to weigh about 0.0001495 × 100^2.962 ≈ 125 g whole. Remember that UCI values must be multiplied by 200 to recover mm and g.

## 5. Growth parameters

Von Bertalanffy and Gompertz parameters, sexes combined, one ring per year (Nash et al. 1994, Table 4, p. 29):

| Area | VB L∞ (mm) | VB K (per year) | VB t0 (years) | Gompertz L∞ (mm) | n |
|---|---|---|---|---|---|
| Hogan Is. | 135.7 | 0.168 | 3.508 | 128.8 | 413 |
| Kent Is. | 127.0 | 0.135 | 2.388 | 120.2 | 356 |
| Babel I. | 146.1 | 0.136 | 2.863 | 130.4 | 414 |
| Cape Barren I. | 139.4 | 0.129 | 2.634 | 129.2 | 344 |
| Waterhouse I. | 121.6 | 0.151 | 2.545 | 112.6 | 303 |

- With two rings per year, K doubles and t0 halves; L∞ is unchanged (Nash et al. 1994, p. 20).
- Females generally have a smaller L∞ and a larger K than males (Nash et al. 1994, p. 26).
- Southern Tasmania (Blubber Head): below 80 mm, A (months) = 1.41 + 0.58 L (mm); above 80 mm, L(t) = 139.7 × [1 − e^(−0.024 (t − 12.1))] with t in months (Prince et al. 1988a, p. 171). Equivalent form in years: L = 20.83 A − 2.46 below 80 mm; K = 0.288 per year, L∞ = 139.7 mm, t0 = 1.01 years above 80 mm (Prince et al. 1988b, p. 78).
- Victoria (tag recapture): K = 0.37, 0.35 and 0.15 per year and L∞ = 152.1, 121.0 and 121.0 mm at Sandpatch, Tullaburga Island and Port Fairy (McShane and Smith 1992, Table 1, p. 1217).

## 6. Mortality

- Total mortality Z (one ring per year): 0.219 (Hogan), 0.250 (Kent), 0.207 (Babel), 0.185 (Cape Barren), 0.265 (Waterhouse) (Nash et al. 1994, Table 8, p. 42).
- Natural mortality is about 0.70 per year in the second year of life and declines with age (Prince et al. 1988b, pp. 79-80).
