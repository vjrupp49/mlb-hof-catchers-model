# ============================================================
# hof_probability_chart.R
#
# Generates the portfolio chart for the HOF Catchers model:
# predicted Hall of Fame probability for a handful of recognizable
# catchers, plus Joe Mauer's blind holdout prediction (his own row
# fully excluded from training, per the model's own sanity check).
#
# Re-runs the same pipeline as "Catchers HOF Model.qmd" rather than
# reading in saved predictions, so the chart always reflects the
# actual fitted model.
# ============================================================

library(dplyr)
library(stringr)
library(janitor)
library(Lahman)
library(tidymodels)
library(readxl)
library(ggplot2)
library(forcats)

# ---- 1) Load + clean AQCS, same as the model notebook ----
AQCS <- read_excel("All Qualified Catchers Stats.xlsx") %>%
  clean_names() %>%
  mutate(
    name_clean = name %>%
      str_replace_all("\\.", "") %>%
      str_replace_all(",", "") %>%
      str_replace_all("'", "") %>%
      str_replace_all("-", " ") %>%
      str_squish() %>%
      str_to_lower()
  )

people_lookup <- Lahman::People %>%
  as_tibble() %>%
  transmute(
    playerID,
    name_clean = paste(nameFirst, nameLast) %>%
      str_replace_all("\\.", "") %>%
      str_replace_all(",", "") %>%
      str_replace_all("'", "") %>%
      str_replace_all("-", " ") %>%
      str_squish() %>%
      str_to_lower()
  )

AQCS <- AQCS %>% left_join(people_lookup, by = "name_clean")

manual_name_map <- tibble::tribble(
  ~name,                  ~playerID,
  "A.J. Pierzynski",      "pierzaj01",
  "B.J. Surhoff",         "surhobj01",
  "Billy Sullivan Jr.",   "sullibi04",
  "Billy Sullivan Sr.",   "sullibi03",
  "Chief Meyers",         "meyerch01",
  "Chief Zimmer",         "zimmech01",
  "Christian Vázquez",    "vazquch01",
  "Eddie Taubensee",      "taubeed01",
  "Gary Sánchez",         "sanchga02",
  "J.T. Realmuto",       "realmjt01",
  "Martín Maldonado",     "maldoma01",
  "Sandy Alomar Jr.",     "alomasa02",
  "Tony Peña Sr.",        "penato01"
)

AQCS <- AQCS %>%
  left_join(manual_name_map, by = "name", suffix = c("", "_manual")) %>%
  mutate(playerID = coalesce(playerID, playerID_manual)) %>%
  select(-playerID_manual) %>%
  filter(name != "Frank Duncan")

# ---- 2) Awards + All-Star features, same as the model notebook ----
allstar_counts <- Lahman::AllstarFull %>%
  as_tibble() %>%
  count(playerID, name = "all_star_selections")

AQCS <- AQCS %>%
  left_join(allstar_counts, by = "playerID") %>%
  mutate(all_star_selections = coalesce(all_star_selections, 0L))

awards <- Lahman::AwardsPlayers %>% as_tibble()

gg_counts  <- awards %>% filter(awardID == "Gold Glove")            %>% count(playerID, name = "gg")
ss_counts  <- awards %>% filter(awardID == "Silver Slugger")        %>% count(playerID, name = "ss")
mvp_awards <- awards %>% filter(awardID == "Most Valuable Player")  %>% count(playerID, name = "mvp_awards")

AQCS <- AQCS %>%
  left_join(gg_counts,  by = "playerID") %>%
  left_join(ss_counts,  by = "playerID") %>%
  left_join(mvp_awards, by = "playerID") %>%
  mutate(gg = coalesce(gg, 0L), ss = coalesce(ss, 0L), mvp_awards = coalesce(mvp_awards, 0L))

# ---- 3) HOF label, same cutoff logic as the model notebook ----
cutoff_last_mlb_year <- 2012

hof_lookup <- Lahman::HallOfFame %>%
  as_tibble() %>%
  group_by(playerID) %>%
  summarise(inducted_any = any(inducted == "Y", na.rm = TRUE), .groups = "drop")

last_mlb_year_lookup <- Lahman::Batting %>%
  as_tibble() %>%
  group_by(playerID) %>%
  summarise(last_mlb_year = max(yearID, na.rm = TRUE), .groups = "drop")

AQCS <- AQCS %>%
  left_join(hof_lookup, by = "playerID") %>%
  left_join(last_mlb_year_lookup, by = "playerID") %>%
  mutate(
    inducted_any = coalesce(inducted_any, FALSE),
    hof_resolved = inducted_any | (!is.na(last_mlb_year) & last_mlb_year <= cutoff_last_mlb_year),
    hof = case_when(
      inducted_any ~ 1L,
      hof_resolved ~ 0L,
      TRUE ~ NA_integer_
    )
  )

# ---- 4) Train/test split + model, same as the model notebook ----
model_features <- c("war","pa","g","w_rc","w_oba","def","fielding","positional",
                     "all_star_selections","gg","ss","mvp_awards")

AQCS_model_ready <- AQCS %>%
  filter(!is.na(hof)) %>%
  select(all_of(model_features), hof, last_mlb_year)

train_cutoff_year <- 2005

train_data <- AQCS_model_ready %>% filter(last_mlb_year <= train_cutoff_year) %>% select(-last_mlb_year) %>%
  mutate(hof = factor(hof, levels = c(0, 1)))

set.seed(4)

hof_recipe <- recipe(hof ~ ., data = train_data) %>%
  step_impute_median(all_predictors()) %>%
  step_normalize(all_predictors())

hof_model <- logistic_reg(penalty = tune(), mixture = 0) %>% set_engine("glmnet")
hof_workflow <- workflow() %>% add_recipe(hof_recipe) %>% add_model(hof_model)

hof_folds <- vfold_cv(train_data, v = 5, repeats = 10, strata = hof)
penalty_grid <- grid_regular(penalty(range = c(-4, 1)), levels = 20)

hof_tune <- tune_grid(hof_workflow, resamples = hof_folds, grid = penalty_grid,
                       metrics = metric_set(roc_auc))
best_penalty <- select_best(hof_tune, metric = "roc_auc")

final_workflow <- finalize_workflow(hof_workflow, best_penalty)
final_fit <- fit(final_workflow, data = train_data)

# ---- 5) Predict every unresolved (still-active-case) catcher ----
predict_data <- AQCS %>% filter(is.na(hof))

predict_results <- predict(final_fit, new_data = predict_data, type = "prob") %>%
  bind_cols(predict_data %>% select(name, playerID)) %>%
  rename(p_hof = .pred_1) %>%
  arrange(desc(p_hof))

# ---- 6) Joe Mauer, predicted fully blind (his row never touches training) ----
joe_id <- "mauerjo01"
joe_row <- AQCS %>% filter(playerID == joe_id)

AQCS_model_mauer <- AQCS %>% filter(!is.na(hof)) %>% filter(playerID != joe_id)

model_df <- AQCS_model_mauer %>%
  select(all_of(model_features), hof, last_mlb_year) %>%
  mutate(hof = factor(hof, levels = c(0, 1)))

train_df_mauer <- model_df %>% filter(last_mlb_year <= train_cutoff_year) %>% select(-last_mlb_year)

set.seed(4)
folds_mauer <- vfold_cv(train_df_mauer, v = 5, repeats = 10, strata = hof)

tuned_mauer <- tune_grid(hof_workflow, resamples = folds_mauer, grid = penalty_grid,
                          metrics = metric_set(roc_auc))
best_pen_mauer <- select_best(tuned_mauer, metric = "roc_auc")

final_wf_mauer  <- finalize_workflow(hof_workflow, best_pen_mauer)
final_fit_mauer <- fit(final_wf_mauer, data = train_df_mauer)

joe_prob <- predict(final_fit_mauer, new_data = joe_row %>% select(all_of(model_features)), type = "prob") %>%
  rename(p_hof = .pred_1) %>%
  mutate(name = joe_row$name, playerID = joe_id) %>%
  select(name, playerID, p_hof)

# ---- 7) Assemble the chart: top unresolved catchers + Joe Mauer's blind test ----
chart_data <- predict_results %>%
  slice_max(p_hof, n = 6) %>%
  bind_rows(joe_prob) %>%
  distinct(playerID, .keep_all = TRUE) %>%
  mutate(
    is_mauer = playerID == joe_id,
    label = if_else(is_mauer, paste0(name, " (blind holdout)"), name)
  ) %>%
  arrange(p_hof)

p <- ggplot(chart_data, aes(x = fct_inorder(label), y = p_hof, fill = is_mauer)) +
  geom_col(width = 0.65) +
  geom_text(aes(label = scales::percent(p_hof, accuracy = 1)),
            hjust = -0.15, size = 3.6, family = "sans", color = "#2B2620") +
  coord_flip(clip = "off") +
  scale_y_continuous(labels = scales::percent_format(), limits = c(0, 1.08), expand = c(0, 0)) +
  scale_fill_manual(values = c(`FALSE` = "#8A9389", `TRUE` = "#C7842A"), guide = "none") +
  labs(
    title = "Predicted Hall of Fame Probability",
    subtitle = "Regularized logistic regression, ridge-penalized, tuned via repeated cross-validation\nJoe Mauer's own row was fully excluded from training before this prediction",
    x = NULL, y = "Predicted probability of induction",
    caption = "Model: Catchers HOF Model.qmd  |  Data: Lahman database + hand-compiled qualified-catcher stats"
  ) +
  theme_minimal(base_size = 12) +
  theme(
    plot.title = element_text(face = "bold", size = 15),
    plot.subtitle = element_text(color = "#5B6169", size = 10, margin = margin(b = 10)),
    plot.caption = element_text(color = "#8A9389", size = 8, margin = margin(t = 10)),
    axis.title.x = element_text(size = 10, color = "#5B6169", margin = margin(t = 8)),
    axis.title.y = element_blank(),
    axis.text = element_text(size = 10.5, color = "#2B2620"),
    panel.grid.minor = element_blank(),
    panel.grid.major.y = element_blank(),
    plot.margin = margin(12, 30, 12, 12)
  )

ggsave("hof_probability_chart.png", p, width = 8, height = 5, dpi = 200, bg = "white")

cat("\nSaved hof_probability_chart.png\n\n")
print(chart_data %>% select(name, p_hof))
