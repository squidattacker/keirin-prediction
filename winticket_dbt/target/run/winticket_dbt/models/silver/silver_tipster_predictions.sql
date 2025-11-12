
  
  create view "winticket"."main_silver"."silver_tipster_predictions__dbt_tmp" as (
    

with rc as (
    select
        race_id,
        state['race_common']['predictions'] as predictions
    from "winticket"."main_bronze"."bronze_racecards"
),
filtered as (
    select
        race_id,
        pred
    from rc
    cross join unnest(predictions) as p(pred)
    where pred['tipsterId']::varchar = 'dsc-00'
)
select
    race_id,
    pred['tipsterId']::varchar as tipster_id,
    pred['name']::varchar as tipster_name,
    idx.generate_series::int as pick_rank,
    list_extract(pred['order'], idx.generate_series)::int as number
from filtered
cross join generate_series(1, array_length(pred['order'])) as idx(generate_series)
  );
