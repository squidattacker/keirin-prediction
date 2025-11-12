

with odds as (
    select
        race_id,
        odds_struct
    from "winticket"."main_bronze"."bronze_racecards"
    cross join unnest(state['race_odds']['exacta']) as o(odds_struct)
),
prep as (
    select
        race_id,
        odds_struct['key'][1]::int as first_number,
        odds_struct['key'][2]::int as second_number,
        odds_struct['odds']::double as odds,
        odds_struct['popularityOrder']::double as popularity_order
    from odds
    where coalesce(odds_struct['absent']::boolean, false) = false
)
select
    race_id,
    first_number,
    min(odds) as min_exacta_odds_first,
    max(odds) as max_exacta_odds_first,
    avg(odds) as avg_exacta_odds_first,
    max(odds) - min(odds) as spread_exacta_odds_first,
    min(popularity_order) as min_exacta_popularity_first,
    avg(popularity_order) as avg_exacta_popularity_first
from prep
group by 1,2