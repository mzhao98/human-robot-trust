DROP DATABASE IF EXISTS sargame;
CREATE DATABASE sargame;
USE sargame;

CREATE TABLE `game`(
    `id` INT(6) UNSIGNED AUTO_INCREMENT PRIMARY KEY,
    `userid` VARCHAR(50) NOT NULL,
    `episode` INT(6) NOT NULL,
    `saving_bool` VARCHAR(1000),
    `victim_pos` VARCHAR(1000),
    `num_step` INT(6),
    `time_spent` VARCHAR(20),
    `trajectory` VARCHAR(10000),
    `created_at` TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    `advice_message` VARCHAR(10000),
    `condition` INT(1),
    `player_score` INT(6),
    `quiz_score` INT(6),
    `survey_key` VARCHAR(20)

);

CREATE TABLE `survey`(
    `id` INT(6) UNSIGNED AUTO_INCREMENT PRIMARY KEY,
    `userid` VARCHAR(50) NOT NULL,
    `episode` INT(6) NOT NULL,
    `condition` INT(1),
    `player_score` INT(6),
    `quiz_score` INT(6),
    `survey_key` VARCHAR(20),
    `question` VARCHAR(1000),
    `response` VARCHAR(1000)

);

