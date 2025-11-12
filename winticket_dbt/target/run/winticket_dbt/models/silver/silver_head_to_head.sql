
  
  create view "winticket"."main_silver"."silver_head_to_head__dbt_tmp" as (
    

with rc as (
    select
        race_id,
        state['race_common']['competitionRecords'] as competition_records
    from "winticket"."main_bronze"."bronze_racecards"
),
flat as (
    select
        race_id,
        comp['playerId']::varchar as player_id,
        comp['opponentId']::varchar as opponent_id,
        coalesce(comp['wins']::int, 0) as wins,
        coalesce(comp['losses']::int, 0) as losses
    from rc
    cross join unnest(competition_records) as c(comp)
),
field_entries as (
    select distinct race_id, player_id
    from "winticket"."main_silver"."silver_race_entries"
)
select
    f.race_id,
    f.player_id,
    sum(f.wins) filter (where fe.player_id is not null) as field_head_to_head_wins,
    sum(f.losses) filter (where fe.player_id is not null) as field_head_to_head_losses,
    sum(f.wins + f.losses) filter (where fe.player_id is not null) as field_head_to_head_races
from flat f
left join field_entries fe
    on f.race_id = fe.race_id
   and f.opponent_id = fe.player_id
group by 1,2
  );
