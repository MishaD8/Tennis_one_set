Tennis API Version - 2.9.4
Welcome to the Tennis API service documentation page. Please see the whole list of techniques listed below for real-time tennis data that will help you create fantastic applications.

Events Types
Method
GET/POST api.api-tennis.com/tennis/?method=get_events
Returns list of supported tournaments types included in your current subscription plan.

Parameters
Parameter	Description
method	API method name
APIkey	Authorization code generated from your api tennis account
Request URL
https://api.api-tennis.com/tennis/?method=get_events&APIkey=!_your_account_APIkey_!
JSON Response
{
	"success": 1,
	"result": [
		{
			"event_type_key": "267",
			"event_type_type": "Atp Doubles"
		},
		{
			"event_type_key": "265",
			"event_type_type": "Atp Singles"
		},
		{
			"event_type_key": "279",
			"event_type_type": "Boys Doubles"
		},
		{
			"event_type_key": "277",
			"event_type_type": "Boys Singles"
		},
		{
			"event_type_key": "282",
			"event_type_type": "Challenger Men Doubles"
		},
		{
			"event_type_key": "281",
			"event_type_type": "Challenger Men Singles"
		},
		{
			"event_type_key": "275",
			"event_type_type": "Challenger Women Doubles"
		},
        ............
   ]
}

Tournaments
Method
GET/POST api.api-tennis.com/tennis/?method=get_tournaments
Returns list of supported tournaments included in your current subscription plan.

Parameters
Parameter	Description
method	API method name
APIkey	Authorization code generated from your api tennis account
Request URL
https://api.api-tennis.com/tennis/?method=get_tournaments&APIkey=!_your_account_APIkey_!
JSON Response
{
	"success": 1,
	"result": [
		{
			"tournament_key": "2833",
			"tournament_name": "Aachen",
			"event_type_key": "281",
			"event_type_type": "Challenger Men Singles"
		},
		{
			"tournament_key": "3872",
			"tournament_name": "Abu Dhabi",
			"event_type_key": "266",
			"event_type_type": "Wta Singles"
		},
		{
			"tournament_key": "2801",
			"tournament_name": "Abu Dhabi",
			"event_type_key": "276",
			"event_type_type": "Exhibition Women"
		},
		{
			"tournament_key": "2655",
			"tournament_name": "Abu Dhabi",
			"event_type_key": "283",
			"event_type_type": "Exhibition Men"
		},
		{
			"tournament_key": "2130",
			"tournament_name": "Acapulco",
			"event_type_key": "266",
			"event_type_type": "Wta Singles"
		},
		{
			"tournament_key": "2131",
			"tournament_name": "Acapulco",
			"event_type_key": "265",
			"event_type_type": "Atp Singles"
		},
        ............
   ]
}

Fixtures
Method
GET/POST api.api-tennis.com/tennis/?method=get_fixtures
Returns tennis fixtures included in your current subscription plan

Parameters
Parameter	Description
method	API method name
APIkey	Authorization code generated from your api tennis account
date_start	Start date (yyyy-mm-dd)
date_stop	Stop date (yyyy-mm-dd)
event_type_key	Event Type Key - if set only matches from specific tennis event type will be returned (Optional)
tournament_key	Tournament Key - if set only matches from specific tennis tournament will be returned (Optional)
tournament_season	Tournament Season - if set only matches from specific tennis tournament season will be returned (Optional)
match_key	Match Key - if set only details from specific tennis match will be returned (Optional)
player_key	Player Key - if set only details from specific tennis player will be returned (Optional)
timezone	The timezone where you want to receive the data in tz format (exemple: America/New_York). Default timezone: Europe/Berlin (Optional)
Request URL
https://api.api-tennis.com/tennis/?method=get_fixtures&APIkey=!_your_account_APIkey_!&date_start=2019-07-24&date_stop=2019-07-24
JSON Response
       {
    "success": 1,
    "result": [
        {
            "event_key": "143104",
            "event_date": "2022-06-17",
            "event_time": "18:00",
            "event_first_player": "M. Navone",
            "first_player_key": "949",
            "event_second_player": "C. Gomez-Herrera",
            "second_player_key": "3474",
            "event_final_result": "-",
            "event_game_result": "-",
            "event_serve": null,
            "event_winner": null,
            "event_status": "",
            "event_type_type": "Challenger Men Singles",
            "tournament_name": "Corrientes Challenger Men",
            "tournament_key": "2646",
            "tournament_round": "",
            "tournament_season": "2022",
            "event_live": "0",
            "event_qualification": "False",
            "event_first_player_logo": null,
            "event_second_player_logo": "https://api.api-tennis.com/logo-tennis/3474_c-gomez-herrera.jpg",
            "pointbypoint": [],
            "scores": []
        },
        {
            "event_key": "143113",
            "event_date": "2022-06-17",
            "event_time": "01:05",
            "event_first_player": "C. Chidekh",
            "first_player_key": "7102",
            "event_second_player": "M. Cassone",
            "second_player_key": "12744",
            "event_final_result": "2 - 0",
            "event_game_result": "-",
            "event_serve": null,
            "event_winner": "First Player",
            "event_status": "Finished",
            "event_type_type": "Itf Men Singles",
            "tournament_name": "ITF M25 Wichita, KS Men",
            "tournament_key": "4195",
            "tournament_round": "",
            "tournament_season": "2022",
            "event_live": "0",
            "event_first_player_logo": null,
            "event_second_player_logo": null,
            "pointbypoint": [
                {
                    "set_number": "Set 1",
                    "number_game": "1",
                    "player_served": "First Player",
                    "serve_winner": "First Player",
                    "serve_lost": null,
                    "score": "1 - 0",
                    "points": [
                        {
                            "number_point": "1",
                            "score": "15 - 0",
                            "break_point": null,
                            "set_point": null,
                            "match_point": null
                        },
                      .........
                    ],
                },
                ...........
              ],
              "scores": [
                {
                    "score_first": "6",
                    "score_second": "4",
                    "score_set": "1"
                },
                {
                    "score_first": "6",
                    "score_second": "2",
                    "score_set": "2"
                }
              ]
          }
      }
  ]
}


Livescore
Method
GET/POST api.api-tennis.com/tennis/?method=get_livescore
Returns tennis now playing events included in your current subscription plan.

Parameters
Parameter	Description
method	API method name
APIkey	Authorization code generated from your api tennis account
event_type_key	Event Type Key - if set only matches from specific tennis event type will be returned (Optional)
tournament_key	Tournament Key - if set only matches from specific tennis tournament will be returned (Optional)
match_key	Match Key - if set only details from specific tennis match will be returned (Optional)
player_key	Player Key - if set only details from specific tennis player will be returned (Optional)
timezone	The timezone where you want to receive the data in tz format (exemple: America/New_York). Default timezone: Europe/Berlin (Optional)
Request URL
https://api.api-tennis.com/tennis/?method=get_livescore&APIkey=!_your_account_APIkey_!
JSON Response
{
    "success": 1,
    "result": [
        {
            "event_key": "143192",
            "event_date": "2022-06-17",
            "event_time": "10:10",
            "event_first_player": "S. Bejlek",
            "first_player_key": "9393",
            "event_second_player": "R. Zarazua",
            "second_player_key": "1805",
            "event_final_result": "0 - 0",
            "event_game_result": "0 - 0",
            "event_serve": "First Player",
            "event_winner": null,
            "event_status": "Set 1",
            "event_type_type": "Itf Women Singles",
            "tournament_name": "ITF W60 Ceska Lipa Women",
            "tournament_key": "4210",
            "tournament_round": "",
            "tournament_season": "2022",
            "event_live": "1",
            "event_first_player_logo": null,
            "event_second_player_logo": "https://api.tennis.com/logo-tennis/1805_r-zarazua.jpg",
            "event_qualification": "False",
            "pointbypoint": [
                {
                    "set_number": "Set 1",
                    "number_game": "1",
                    "player_served": "First Player",
                    "serve_winner": "First Player",
                    "serve_lost": null,
                    "score": "1 - 0",
                    "points": [
                        {
                            "number_point": "1",
                            "score": "15 - 0",
                            "break_point": null,
                            "set_point": null,
                            "match_point": null
                        },
                        .............
                    ],
                },
                .............
            ],
            "scores": [
                {
                    "score_first": "5",
                    "score_second": "5",
                    "score_set": "1"
                }
            ]
        }
    ]
    .......
}

H2H (Head to Head)
Method
GET/POST api.api-tennis.com/tennis/?method=get_H2H
Returns the last games between submiteted players and the last games of each player

Parameters
Parameter	Description
action	API method name
APIkey	Authorization code generated from your api tennis account
first_player_key	First player Key
second_player_key	Second player Key
Request URL
https://api.api-tennis.com/tennis/?method=get_H2H&APIkey=!_your_account_APIkey_!&first_player_key=30&second_player_key=5
JSON Response
{
	"success": 1,
	"result": {
		"H2H": [],
		"firstPlayerResults": [
			{
				"event_key": "112163",
				"event_date": "2022-05-11",
				"event_time": "15:00",
				"event_first_player": "Cervantes Tomas/ Ferrer Adria",
				"first_player_key": "2616",
				"event_second_player": "Kravchenko/ Reymond",
				"second_player_key": "2316",
				"event_final_result": "0 - 2",
				"event_game_result": "-",
				"event_serve": null,
				"event_winner": "Second Player",
				"event_status": "Finished",
				"event_type_type": "Itf Men Doubles",
				"tournament_name": "ITF M15 Ulcinj Men",
				"tournament_key": "4561",
				"tournament_round": "ITF M15 Ulcinj Men - 1/8-finals",
				"tournament_season": "2022",
				"event_live": "0",
				"event_first_player_logo": null,
				"event_second_player_logo": null
			},
			....
			secondPlayerResults": [
			{
				"event_key": "94804",
				"event_date": "2022-05-11",
				"event_time": "15:10",
				"event_first_player": "Lopez San Martin/ Rincon",
				"first_player_key": "2139",
				"event_second_player": "Regas/ Vasershtein",
				"second_player_key": "2617",
				"event_final_result": "2 - 0",
				"event_game_result": "-",
				"event_serve": null,
				"event_winner": "First Player",
				"event_status": "Finished",
				"event_type_type": "Itf Men Doubles",
				"tournament_name": "ITF M15 Valldoreix Men",
				"tournament_key": "3855",
				"tournament_round": "ITF M15 Valldoreix Men - Quarter-finals",
				"tournament_season": "2022",
				"event_live": "0",
				"event_first_player_logo": null,
				"event_second_player_logo": null
			},
			....
  }

Standings
Method
GET/POST api.api-tennis.com/tennis/?method=get_standings
Returns standings for tennis tournaments included in your current subscription plan.

Parameters
Parameter	Description
action	API method name
APIkey	Authorization code generated from your api tennis account
event_type	'ATP' or 'WTA'
Request URL
https://api.api-tennis.com/tennis/?method=get_standings&event_type=WTA&APIkey=!_your_account_APIkey_!
JSON Response
{
	"success": 1,
	"result": [
		{
			"place": "1",
			"player": "Iga Swiatek",
			"player_key": "1910",
			"league": "WTA",
			"movement": "same",
			"country": "Poland",
			"points": "8501"
		},
		{
			"place": "2",
			"player": "Anett Kontaveit",
			"player_key": "2388",
			"league": "WTA",
			"movement": "same",
			"country": "Estonia",
			"points": "4476"
		},
		{
			"place": "3",
			"player": "Maria Sakkari",
			"player_key": "2076",
			"league": "WTA",
			"movement": "down",
			"country": "Greece",
			"points": "4190"
		},
		.....
}

Players
Method
GET/POST api.api-tennis.com/tennis/?method=get_players
Returns tennis players profile.

Parameters
Parameter	Description
action	API method name
APIkey	Authorization code generated from your api tennis account
player_key	Player internal code
tournament_key	Tournament internal code
Request URL
https://api.api-tennis.com/tennis/?method=get_players&player_key=137&APIkey=!_your_account_APIkey_!
JSON Response
{
	"success": 1,
	"result": [
		{
			"player_key": "1905",
			"player_name": "N. Djokovic",
			"player_country": "Serbia",
			"player_bday": "22.05.1987",
			"player_logo": "https://api.api-tennis.com/logo-tennis/1905_n-djokovic.jpg",
			"stats": [
				{
					"season": "2021",
					"type": "doubles",
					"rank": "255",
					"titles": "0",
					"matches_won": "6",
					"matches_lost": "4",
					"hard_won": "2",
					"hard_lost": "2",
					"clay_won": "",
					"clay_lost": "",
					"grass_won": "3",
					"grass_lost": "0"
				},
				{
					"season": "2020",
					"type": "doubles",
					"rank": "158",
					"titles": "0",
					"matches_won": "2",
					"matches_lost": "1",
					"hard_won": "2",
					"hard_lost": "1",
					"clay_won": "",
					"clay_lost": "",
					"grass_won": "",
					"grass_lost": ""
				},
        ........
     ]
 }

Odds
Method
GET/POST api.api-tennis.com/tennis?method=get_odds
Returns odds for tennis matches included in your current subscription plan.

Parameters
Parameter	Description
method	API method name
APIkey	Authorization code generated from your api tennis account
date_start	Start date (yyyy-mm-dd)
date_stop	Stop date (yyyy-mm-dd)
event_type_key	Event Type Key - if set only matches from specific tennis event type will be returned (Optional)
tournament_key	Tournament Key - if set only matches from specific tennis tournament will be returned (Optional)
match_key	Match Key - if set only details from specific tennis match will be returned (Optional)
Request URL
https://api.api-tennis.com/tennis/?method=get_odds&match_key=159923&APIkey=!_your_account_APIkey_!
JSON Response
{
	"success": 1,
	"result": {
		"159923": {
			"Home/Away": {
				"Home": {
					"bwin": "2.40",
					"bet365": "2.50",
					"Betsson": "2.45",
					"1xbet": "2.50",
					"Sportingbet": "2.40",
					"Betcris": "2.43"
				},
				"Away": {
					"bwin": "1.48",
					"bet365": "1.50",
					"Betsson": "1.48",
					"1xbet": "1.51",
					"Sportingbet": "1.48",
					"Betcris": "1.48"
				}
			},
			"Correct Score 1st Half": {
				"6:0": {
					"bet365": "51.00",
					"1xbet": "51.00"
				},
				"6:1": {
					"bet365": "19.00",
					"1xbet": "19.00"
				},
				"6:2": {
					"bet365": "15.00",
					"1xbet": "15.00"
				},
				"0:6": {
					"bet365": "26.00",
					"1xbet": "26.00"
				},
				"1:6": {
					"bet365": "12.00",
					"1xbet": "12.00"
				},
				"2:6": {
					"bet365": "7.00",
					"1xbet": "7.00"
				},
				"4:6": {
					"bet365": "5.50",
					"1xbet": "5.50"
				},
				"3:6": {
					"bet365": "7.00",
					"1xbet": "7.00"
				},
				"6:3": {
					"bet365": "8.00",
					"1xbet": "8.00"
				},
				"6:4": {
					"bet365": "9.50",
					"1xbet": "9.50"
				},
				"7:5": {
					"bet365": "19.00",
					"1xbet": "19.00"
				},
				"7:6": {
					"bet365": "12.00",
					"1xbet": "12.00"
				},
				"5:7": {
					"bet365": "15.00",
					"1xbet": "15.00"
				},
				"6:7": {
					"bet365": "10.00",
					"1xbet": "10.00"
				}
			},
			"Home/Away (1st Set)": {
				"Home": {
					"bet365": "2.37",
					"1xbet": "2.36"
				},
				"Away": {
					"bet365": "1.53",
					"1xbet": "1.55"
				}
			},
			"Set Betting": {
				"2:0": {
					"bwin": "3.90",
					"bet365": "3.75",
					"1xbet": "3.74",
					"Sportingbet": "3.90",
					"Betcris": "3.80"
				},
				"2:1": {
					"bwin": "5.50",
					"bet365": "6.00",
					"1xbet": "6.00",
					"Sportingbet": "5.50",
					"Betcris": "5.20"
				},
				"0:2": {
					"bwin": "2.10",
					"bet365": "2.10",
					"1xbet": "2.10",
					"Sportingbet": "2.10",
					"Betcris": "2.00"
				},
				"1:2": {
					"bwin": "4.20",
					"bet365": "4.75",
					"1xbet": "4.74",
					"Sportingbet": "4.20",
					"Betcris": "4.10"
				}
			},
			"Win In Straigh Sets (Player 1)": {
				"Yes": {
					"bet365": "3.75"
				},
				"No": {
					"bet365": "1.25"
				}
			},
			"Win In Straigh Sets (Player 2)": {
				"Yes": {
					"bet365": "2.10"
				},
				"No": {
					"bet365": "1.66"
				}
			}
		}
	}

Live Odds
Method
GET/POST api.api-tennis.com/tennis/?method=get_live_odds
Returns tennis live odds for live matches.

Parameters
Parameter	Description
method	API method name
APIkey	Authorization code generated from your api tennis account
event_type_key	Event Type Key - if set only matches from specific tennis event type will be returned (Optional)
tournament_key	Tournament Key - if set only matches from specific tennis tournament will be returned (Optional)
match_key	Match Key - if set only details from specific tennis match will be returned (Optional)
player_key	Player Key - if set only details from specific tennis player will be returned (Optional)
timezone	The timezone where you want to receive the data in tz format (exemple: America/New_York). Default timezone: Europe/Berlin (Optional)
Request URL
https://api.api-tennis.com/tennis/?method=get_live_odds&APIkey=!_your_account_APIkey_!
JSON Response
{
    "success": 1,
    "result": {
        "11976653": {
            "event_key": 11976653,
            "event_date": "2024-08-22",
            "event_time": "08:45",
            "first_player_key": 69252,
            "second_player_key": 73380,
            "event_game_result": "30 - 30",
            "event_serve": "First Player",
            "event_winner": null,
            "event_status": "Set 2",
            "event_type_type": "Itf Women Doubles",
            "tournament_name": "ITF W35 Kunshan Women",
            "tournament_key": 11556,
            "tournament_round": "ITF W35 Kunshan Women - Quarter-finals",
            "tournament_season": "2024",
            "event_live": "08:45",
            "event_first_player_logo": null,
            "event_second_player_logo": null,
            "event_qualification": "False",
            "live_odds": [
                {
                    "odd_name": "Set 1 to Break Serve",
                    "suspended": "Yes",
                    "type": "1/Yes",
                    "value": "1.125",
                    "handicap": null,
                    "upd": "2024-08-22 09:15:10"
                },
                {
                    "odd_name": "Set 1 to Break Serve",
                    "suspended": "Yes",
                    "type": "2/Yes",
                    "value": "1.111",
                    "handicap": null,
                    "upd": "2024-08-22 09:15:10"
                },
                ...........
              ]
          },
          .........
      }
}

## Tennis Backend API-Tennis.com Integration

The tennis backend system now provides comprehensive integration with API-Tennis.com through the following endpoints:

### API-Tennis Integration Status
GET /api/api-tennis/status
Returns the current status of the API-Tennis.com integration, including connectivity and configuration status.

### Tournament Data
GET /api/api-tennis/tournaments
Retrieves all available tournaments from API-Tennis.com with filtering for professional ATP/WTA events only.

### Match Data
GET /api/api-tennis/matches
Parameters:
- include_live (boolean, default: true): Include live matches
- days_ahead (integer, default: 2): Number of days ahead to fetch

Retrieves current and upcoming matches from API-Tennis.com with comprehensive match information.

### Player-Specific Matches
GET /api/api-tennis/player/{player_name}/matches
Parameters:
- days_ahead (integer, default: 30): Number of days ahead to search

Retrieves all matches for a specific player from API-Tennis.com.

### Match Odds
GET /api/api-tennis/match/{match_id}/odds
Retrieves betting odds for a specific match from API-Tennis.com.

### Enhanced Data Collection
GET /api/api-tennis/enhanced
Parameters:
- days_ahead (integer, default: 2): Number of days ahead to fetch

Retrieves comprehensive match data using the Enhanced API-Tennis collector, which combines multiple data sources.

### Connection Testing
GET /api/api-tennis/test-connection
Tests the API-Tennis.com connection and API key validity.

### Cache Management
POST /api/api-tennis/clear-cache
Requires API key authentication. Clears the API-Tennis.com local cache.

### Configuration Requirements

To use API-Tennis.com integration, set the following environment variable:
```
API_TENNIS_KEY=your_api_tennis_key_here
```

The integration includes:
- Automatic rate limiting (50 requests per minute)
- Local caching (15-minute duration)
- Professional tournament filtering (ATP/WTA only)
- Data normalization to Universal Collector format
- Comprehensive error handling