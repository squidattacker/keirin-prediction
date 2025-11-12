

with rr as (
    select
        race_id,
        state['race_common']['results'] as results
    from "winticket"."main_bronze"."bronze_race_results"
)
select
    race_id,
    result['playerId']::varchar as player_id,
    result['order']::int as finish_order,
    result['factor']::varchar as finish_factor,
    result['finalHalfRecord']::varchar as final_half_record
from rr
cross join unnest(results) as r(result)