{{ config(materialized='table') }}

with base as (
    select
        e.*,
        m.schedule_date,
        m.schedule_day,
        m.distance_m,
        m.weather,
        m.wind_speed,
        m.entries_number,
        m.race_class,
        m.race_phase
    from {{ ref('silver_race_entries') }} e
    left join {{ ref('silver_race_metadata') }} m using (race_id)
),
odds as (
    select
        race_id,
        first_number,
        min_exacta_odds_first,
        max_exacta_odds_first,
        avg_exacta_odds_first,
        spread_exacta_odds_first,
        min_exacta_popularity_first,
        avg_exacta_popularity_first
    from {{ ref('silver_exacta_odds') }}
),
h2h as (
    select
        race_id,
        player_id,
        coalesce(field_head_to_head_wins, 0) as field_head_to_head_wins,
        coalesce(field_head_to_head_losses, 0) as field_head_to_head_losses,
        coalesce(field_head_to_head_races, 0) as field_head_to_head_races
    from {{ ref('silver_head_to_head') }}
),
picks as (
    select
        race_id,
        number,
        tipster_id,
        tipster_name,
        pick_rank
    from {{ ref('silver_tipster_predictions') }}
),
results as (
    select * from {{ ref('silver_race_results') }}
)
select
    b.race_id,
    b.cup_id,
    b.race_number,
    b.day_index,
    b.venue_slug,
    b.player_id,
    b.player_name,
    b.number,
    b.bracket_number,
    b.is_absent,
    b.schedule_date,
    b.schedule_day,
    b.distance_m,
    b.weather,
    b.wind_speed,
    b.entries_number,
    b.race_class,
    b.race_phase,
    b.age,
    b.player_class,
    b.player_group,
    b.region_id,
    b.recent_races,
    b.recent_avg_order,
    b.recent_win_rate,
    b.recent_top3_rate,
    b.venue_races,
    b.venue_avg_order,
    b.venue_win_rate,
    b.venue_top3_rate,
    coalesce(b.race_point, 0) as racePoint,
    coalesce(b.first_rate, 0) as firstRate,
    coalesce(b.second_rate, 0) as secondRate,
    coalesce(b.third_rate, 0) as thirdRate,
    coalesce(b.ex_spurt_pct, 0) as exSpurt,
    coalesce(b.ex_thrust_pct, 0) as exThrust,
    coalesce(b.ex_split_pct, 0) as exSplit,
    coalesce(b.ex_compete_pct, 0) as exCompete,
    coalesce(b.ex_leftbehind_pct, 0) as exLeftBehind,
    coalesce(b.line_first_pct, 0) as lineFirstPct,
    coalesce(b.line_single_pct, 0) as lineSinglePct,
    coalesce(o.min_exacta_odds_first, 0) as min_exacta_odds_first,
    coalesce(o.max_exacta_odds_first, 0) as max_exacta_odds_first,
    coalesce(o.avg_exacta_odds_first, 0) as avg_exacta_odds_first,
    coalesce(o.spread_exacta_odds_first, 0) as spread_exacta_odds_first,
    coalesce(o.min_exacta_popularity_first, 0) as min_exacta_popularity_first,
    coalesce(o.avg_exacta_popularity_first, 0) as avg_exacta_popularity_first,
    h.field_head_to_head_wins,
    h.field_head_to_head_losses,
    h.field_head_to_head_races,
    coalesce(p.tipster_id, '{{ var('target_tipster_id') }}') as tipster_id,
    coalesce(p.tipster_name, 'N/A') as tipster_name,
    p.pick_rank,
    r.finish_order,
    case
        when r.finish_order = 1 then 1
        when r.finish_order is null then null
        else 0
    end as is_winner
from base b
left join odds o on b.race_id = o.race_id and b.number = o.first_number
left join h2h h on b.race_id = h.race_id and b.player_id = h.player_id
left join picks p on b.race_id = p.race_id and b.number = p.number
left join results r on b.race_id = r.race_id and b.player_id = r.player_id
