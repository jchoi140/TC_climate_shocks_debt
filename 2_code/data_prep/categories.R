# Country subsets:
emde <- c(
    "AFG", "ALB", "DZA", "AGO", "ATG", "ARG", "ARM", "AZE", "BGD", "BRB", 
    "BLR", "BLZ", "BEN", "BTN", "BOL", "BIH", "BWA", "BRA", "BGR", "BFA",
    "BDI", "CPV", "KHM", "CMR", "CAF", "TCD", "CHL", "CHN", "COL", "COM", 
    "COD", "COG", "CRI", "CIV", "HRV", "DJI", "DMA", "DOM", "ECU", "EGY", 
    "SLV", "GNQ", "ERI", "SWZ", "ETH", "FJI", "GAB", "GMB", "GEO", "GHA", 
    "GRD", "GTM", "GIN", "GNB", "GUY", "HTI", "HND", "IND", "IDN", "IRN", 
    "IRQ", "JAM", "JOR", "KAZ", "KEN", "KIR", "XKX", "KWT", "KGZ", "LAO", 
    "LBN", "LSO", "LBR", "LBY", "MDG", "MWI", "MYS", "MDV", "MLI", "MHL", 
    "MRT", "MUS", "MEX", "FSM", "MDA", "MNG", "MNE", "MAR", "MOZ", "MMR", 
    "NAM", "NRU", "NPL", "NIC", "NER", "NGA", "MKD", "OMN", "PAK", "PLW", 
    "PAN", "PNG", "PRY", "PER", "PHL", "POL", "QAT", "ROU", "RUS", "RWA", 
    "WSM", "STP", "SAU", "SEN", "SRB", "SYC", "SLE", "SLB", "SOM", "ZAF", 
    "SSD", "LKA", "KNA", "LCA", "VCT", "SDN", "SUR", "SYR", "TJK", "TZA", 
    "THA", "TLS", "TGO", "TON", "TTO", "TUN", "TUR", "TKM", "TUV", "UGA", 
    "UKR", "ARE", "URY", "UZB", "VUT", "VEN", "VNM", "YEM", "ZMB", "ZWE"
)

HIPC <- c(
    "AGO", "BEN", "BOL", "BFA", "BDI", "CMR", "CAF", "TCD", "COM", "COG", 
    "CIV", "COD", "ETH", "GMB", "GHA", "GIN", "GNB", "GUY", "HND", "KEN", 
    "LAO", "LBR", "MDG", "MWI", "MLI", "MRT", "MOZ", "MMR"
)

oecd <- c(
    "AUS", "AUT", "BEL", "CAN", "CHL", "COL", "CRI", "CZE", "DNK", "EST", "FIN", "FRA", 
    "DEU", "GRC", "HUN", "ISL", "IRL", "ISR", "ITA", "JPN", "KOR", "LVA", 
    "LTU", "LUX", "MEX", "NLD", "NZL", "NOR", "POL", "PRT", "SVK", "SVN", 
    "ESP", "SWE", "CHE", "TUR", "GBR", "USA"
)

r = read.csv(here::here("1_input", "credit_ratings", "rating_df_v2.csv"))
rated = r %>% filter(!is.na(score_total)) %>% select(ISO) %>% distinct()
rated = rated$ISO

non_rated = r %>% filter(is.na(score_total) & !(ISO %in% rated) & sovereign==1) %>% select(ISO) %>% distinct()
non_rated = non_rated$ISO

eurozone <- c(
    "AUT", "BEL", "HRV", "CYP", "EST", "FIN", "FRA", "DEU", "GRC", "IRL", 
    "ITA", "LVA", "LTU", "LUX", "MLT", "NLD", "PRT", "SVK", "SVN", "ESP"
)

lowincome <- c(
    "AFG", "BEN", "BFA", "BDI", "CAF", "TCD", "COM", "COD", "ERI", "ETH", 
    "GMB", "GIN", "GNB", "HTI", "PRK", "LBR", "MDG", "MWI", "MLI", "MOZ", 
    "NER", "RWA", "SLE", "SOM", "SSD", "SDN", "TJK", "TZA", "TGO", "UGA", 
    "ZMB", "ZWE", "ZAR", "KIR", "TUV", "YEM"
)

lowmidincome <- c(
    "AGO", "ALB", "ARM", "AZE", "BLR", "BLZ", "BOL", "BTN", "CPV", "COL", 
    "CIV", "CUB", "DJI", "DOM", "ECU", "EGY", "SLV", "FJI", "GEO", "GTM", 
    "GUY", "HND", "IDN", "IND", "IRN", "IRQ", "JAM", "JOR", "KAZ", "KGZ", 
    "LAO", "LBN", "MDA", "MKD", "MNG", "MAR", "MMR", "NAM", "NIC", "NGA", 
    "PSE", "PNG", "PRY", "PER", "PHL", "MDG", "MDV", "ROU", "ROM", "RUS", 
    "WSM", "STP", "SEN", "SRB", "SYR", "TJK", "THA", "TLS", "TKM", "TON", 
    "TUN", "TUR", "UKR", "UZB", "VNM", "VUT", "VEN", "ZMB", "BGD", "CMR", 
    "COG", "GHA", "KEN", "KHM", "LSO", "NPL", "PAK", "SLB", "DMA", "FSM", 
    "MRT", "SWZ"
)

upmidincome <- c(
    "ARG", "AZE", "BRB", "BRA", "CHN", "COL", "CRI", "CUB", "DOM", "DZA", 
    "ECU", "FJI", "GAB", "GUY", "IRN", "IRQ", "JAM", "JOR", "KAZ", "LBN", 
    "LBY", "LKA", "MYS", "MHL", "MEX", "MNG", "MNE", "NAM", "PAN", "PER", 
    "PHL", "PRY", "ROU", "RUS", "SRB", "THA", "TUN", "TUR", "TKM", "UKR", 
    "URY", "VEN", "WSM", "ZAF", "BGR", "BIH", "BWA", "LCA", "MUS", "VCT", 
    "ASM", "GNQ", "GRD", "NRU", "SUR"
)

highincome <- c(
    "AUS", "AUT", "BEL", "BHR", "BRN", "CAN", "CHE", "CHL", "CYP", "CZE", 
    "DEU", "DNK", "ESP", "EST", "FIN", "FRA", "GBR", "GRC", "HKG", "HRV", 
    "HUN", "IRL", "ISL", "ISR", "ITA", "JPN", "KOR", "KWT", "LIE", "LTU", 
    "LUX", "LVA", "MAC", "MLT", "NLD", "NOR", "NZL", "OMN", "POL", "PRT", 
    "QAT", "SAU", "SGP", "SVK", "SVN", "SWE", "TWN", "USA", "ABW", "AND", 
    "BMU", "CYM", "GIB", "ARE", "ATG", "BHS", "GRL", "GUM", "IMN", "KNA", 
    "MAF", "MCO", "MNP", "NCL", "PLW", "PRI", "PYF", "SMR", "SYC", "TCA", 
    "TTO", "VGB", "VIR"
)

sis <- c(
  "ATG","BHS","BRB","BLZ","CPV","COM","COK","CUB","DMA","DOM",
  "FJI","GRD","GNB","GUY","HTI","JAM","KIR","MDV","MHL","FSM",
  "MUS","NRU","NIU","PLW","PNG","WSM","STP","SGP","KNA","LCA",
  "VCT","SYC","SLB","SUR","TLS","TON","TTO","TUV","VUT"
)
# COK (Cook Islands) and NIU (Niue) are not UN members but are included (self-governing, in AOSIS)

sids_associate_iso3 <- c(
  "ASM","AIA","ABW","BMU","VGB","CYM","MNP","CUW","PYF","GLP",
  "GUM","MTQ","MSR","NCL","PRI","SXM","TCA","VIR"
)

sids <- c(
    "ATG", "BHS", "BRB", "BLZ", "CPV", "COM", "COK", "CUB", "DMA", "DOM",
    "FJI", "GRD", "GNB", "GUY", "HTI", "JAM", "KIR", "MDV", "MHL", "FSM",
    "MUS", "NRU", "NIU", "PLW", "PNG", "WSM", "STP", "KNA", "LCA",
    "VCT", "SYC", "SLB", "SUR", "TLS", "TON", "TTO", "TUV", "VUT",
    "MDG", "LKA" # Madagascar, Sri Lanka: larger, but officially SIDS
)

sids_carib <- c(
    "ATG", "BHS", "BRB", "BLZ", "DMA", "DOM", "GRD", 
    "HTI", "JAM", "KNA", "LCA", "VCT", "TTO"
)

sids_pacific <- c(
    "MHL", "FSM", "NRU", "PLW", "PNG", "SLB", "TLS", "TUV", "VUT", "KIR", "TON", "WSM", "FJI"
)
