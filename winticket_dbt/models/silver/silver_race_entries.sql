{{ config(materialized='view') }}

with rc as (
    select
        race_id,
        cup_id,
        race_number,
        day_index,
        venue_slug,
        state['race_common']['entries'] as entries,
        state['race_common']['records'] as records,
        state['race_common']['players'] as players
    from {{ ref('bronze_racecards') }}
),
entries as (
    select
        race_id,
        entry['playerId']::varchar as player_id,
        entry['number']::int as number,
        entry['bracketNumber']::int as bracket_number,
        entry['absent']::boolean as is_absent,
        entry
    from rc
    cross join unnest(entries) as e(entry)
),
records as (
    select
        race_id,
        rec['playerId']::varchar as player_id,
        rec
    from rc
    cross join unnest(records) as r(rec)
),
players as (
    select
        race_id,
        pl['id']::varchar as player_id,
        pl
    from rc
    cross join unnest(players) as p(pl)
),
recent_form as (
    select
        race_id,
        rec['playerId']::varchar as player_id,
        count(*)::int as recent_races,
        avg(rr['order']::double)::double as recent_avg_order,
        avg(case when rr['order'] = 1 then 1 else 0 end)::double as recent_win_rate,
        avg(case when rr['order'] <= 3 then 1 else 0 end)::double as recent_top3_rate
    from rc
    cross join unnest(records) as r(rec)
    cross join unnest(r.rec['latestCupResults']) as cup(cup_struct)
    cross join unnest(cup.cup_struct['raceResults']) as rr(rr)
    group by 1,2
),
venue_form as (
    select
        race_id,
        rec['playerId']::varchar as player_id,
        count(*)::int as venue_races,
        avg(rr['order']::double)::double as venue_avg_order,
        avg(case when rr['order'] = 1 then 1 else 0 end)::double as venue_win_rate,
        avg(case when rr['order'] <= 3 then 1 else 0 end)::double as venue_top3_rate
    from rc
    cross join unnest(records) as r(rec)
    cross join unnest(r.rec['latestVenueResults']) as cup(cup_struct)
    cross join unnest(cup.cup_struct['raceResults']) as rr(rr)
    group by 1,2
)
select
    e.race_id,
    rc.cup_id,
    rc.race_number,
    rc.day_index,
    rc.venue_slug,
    e.player_id,
    e.number,
    e.bracket_number,
    e.is_absent,
    r.rec['racePoint']::double as race_point,
    r.rec['firstRate']::double as first_rate,
    r.rec['secondRate']::double as second_rate,
    r.rec['thirdRate']::double as third_rate,
    r.rec['gearRatio']::double as gear_ratio,
    r.rec['predictionMark']::int as prediction_mark,
    r.rec['comment']::varchar as prediction_comment,
    r.rec['style']::varchar as running_style,
    r.rec['exSpurt']['percentage']::double as ex_spurt_pct,
    r.rec['exThrust']['percentage']::double as ex_thrust_pct,
    r.rec['exSplitLine']['percentage']::double as ex_split_pct,
    r.rec['exCompete']['percentage']::double as ex_compete_pct,
    r.rec['exLeftBehind']['percentage']::double as ex_leftbehind_pct,
    r.rec['linePositionFirst']['firstPercentage']::double as line_first_pct,
    r.rec['lineSingleHorseman']['firstPercentage']::double as line_single_pct,
    p.pl['name']::varchar as player_name,
    p.pl['prefecture']::varchar as prefecture,
    p.pl['age']::int as age,
    p.pl['class']::int as player_class,
    p.pl['group']::int as player_group,
    p.pl['regionId']::varchar as region_id,
    coalesce(f.recent_races, 0) as recent_races,
    coalesce(f.recent_avg_order, 0) as recent_avg_order,
    coalesce(f.recent_win_rate, 0) as recent_win_rate,
    coalesce(f.recent_top3_rate, 0) as recent_top3_rate,
    coalesce(v.venue_races, 0) as venue_races,
    coalesce(v.venue_avg_order, 0) as venue_avg_order,
    coalesce(v.venue_win_rate, 0) as venue_win_rate,
    coalesce(v.venue_top3_rate, 0) as venue_top3_rate
from entries e
join rc on e.race_id = rc.race_id
left join records r on e.race_id = r.race_id and e.player_id = r.player_id
left join players p on e.race_id = p.race_id and e.player_id = p.player_id
left join recent_form f on e.race_id = f.race_id and e.player_id = f.player_id
left join venue_form v on e.race_id = v.race_id and e.player_id = v.player_id
