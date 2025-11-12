
  
  create view "winticket"."main_silver"."silver_race_metadata__dbt_tmp" as (
    

with rc as (
    select
        race_id,
        cup_id,
        race_number,
        day_index,
        venue_slug,
        state['race_common']['race'] as race_struct,
        state['race_common']['schedule'] as schedule_struct
    from "winticket"."main_bronze"."bronze_racecards"
)
select
    race_id,
    cup_id,
    race_number,
    day_index,
    venue_slug,
    schedule_struct['date']::varchar as schedule_date,
    schedule_struct['day']::int as schedule_day,
    race_struct['distance']::int as distance_m,
    race_struct['class']::varchar as race_class,
    race_struct['raceType3']::varchar as race_phase,
    race_struct['weather']::varchar as weather,
    race_struct['windSpeed']::varchar as wind_speed,
    race_struct['entriesNumber']::int as entries_number,
    race_struct['startAt']::bigint as start_at_epoch,
    race_struct['closeAt']::bigint as close_at_epoch
from rc
  );
