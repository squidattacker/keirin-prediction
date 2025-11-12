{{ config(materialized='view') }}

with rc as (
    select
        race_id,
        state['race_common']['predictions'] as predictions
    from {{ ref('bronze_racecards') }}
),
filtered as (
    select
        race_id,
        pred
    from rc
    cross join unnest(predictions) as p(pred)
    where pred['tipsterId']::varchar = '{{ var('target_tipster_id') }}'
)
select
    race_id,
    pred['tipsterId']::varchar as tipster_id,
    pred['name']::varchar as tipster_name,
    idx.generate_series::int as pick_rank,
    list_extract(pred['order'], idx.generate_series)::int as number
from filtered
cross join generate_series(1, array_length(pred['order'])) as idx(generate_series)
