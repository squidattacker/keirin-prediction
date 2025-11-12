
  
  create view "winticket"."main_bronze"."bronze_race_results__dbt_tmp" as (
    

select
  meta['raceId']::varchar as race_id,
  meta['cupId']::varchar as cup_id,
  meta['raceNumber']::int as race_number,
  meta['index']::int as day_index,
  meta['venueSlug']::varchar as venue_slug,
  meta['capturedAt']::varchar as captured_at,
  meta,
  state
from read_json_auto('/workspace/app/data/winticket_lake/bronze/results/*.json')
  );
