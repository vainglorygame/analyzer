SELECT
hero.name,
role.name,
(SELECT name from item WHERE id=(SELECT SUBSTRING_INDEX(items, ',', 1))) AS item_1,
(SELECT name from item WHERE id=(SELECT SUBSTRING_INDEX(SUBSTRING_INDEX(items, ',', 2), ',', -1))) AS item_2,
(SELECT name from item WHERE id=(SELECT SUBSTRING_INDEX(SUBSTRING_INDEX(items, ',', 3), ',', -1))) AS item_3,
(SELECT name from item WHERE id=(SELECT SUBSTRING_INDEX(SUBSTRING_INDEX(items, ',', 4), ',', -1))) AS item_4,
(SELECT name from item WHERE id=(SELECT SUBSTRING_INDEX(SUBSTRING_INDEX(items, ',', 5), ',', -1))) AS item_5,
(SELECT name from item WHERE id=(SELECT SUBSTRING_INDEX(SUBSTRING_INDEX(items, ',', 6), ',', -1))) AS item_6,
COUNT(*) AS cnt
FROM participant
JOIN participant_stats ON participant_stats.participant_api_id=participant.api_id
JOIN role ON participant.role_id=role.id
JOIN hero ON participant.hero_id=hero.id
GROUP BY hero.name, role.name, item_1, item_2, item_3, item_4, item_5, item_6
ORDER BY cnt DESC
