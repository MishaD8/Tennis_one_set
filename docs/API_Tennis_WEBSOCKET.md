WebSocket Tennis Version - 1.2.1
Welcome to the Tennis API Websocket documentation page. Please see the whole list of techniques listed below for real-time tennis data that will help you create fantastic applications.

Live Events and Point By Point
Method
WSS wss.api-tennis.com/live
After we have connected to it, it will push the client every time an event appears in the live score and point by point.

Parameters
Parameter	Description
APIkey	Authorization code generated from your api tennis account
tournament_key	Tournament Key - if set only matches from specific tennis tournament will be returned (Optional)
match_key	Match Key - if set only details from specific tennis match will be returned (Optional)
player_key	Player Key - if set only details from specific tennis player will be returned (Optional)
timezone	The timezone where you want to receive the data in tz format (exemple: America/New_York). Default timezone: Europe/Berlin (Optional)
JSON Message Received
{
      "event_key": 11997372,
      "event_date": "2024-11-07",
      "event_time": "09:10",
      "event_first_player": "P. Verbin",
      "first_player_key": 13391,
      "event_second_player": "M. Kamrowski",
      "second_player_key": 15215,
      "event_final_result": "0 - 0",
      "event_game_result": "0 - 0",
      "event_serve": "Second Player",
      "event_winner": null,
      "event_status": "Set 1",
      "event_type_type": "Itf Men Singles",
      "tournament_name": "ITF M15 Sharm ElSheikh 15 Men",
      "tournament_key": 8153,
      "tournament_round": null,
      "tournament_season": "2024",
      "event_live": "1",
      "event_first_player_logo": null,
      "event_second_player_logo": null,
      "event_qualification": null,
      "pointbypoint": [
          {
              "set_number": "Set 1",
              "number_game": "1",
              "player_served": "First Player",
              "serve_winner": "Second Player",
              "serve_lost": "First Player",
              "score": "0 - 1",
              "points": [
                  {
                      "number_point": "1",
                      "score": "15 - 0",
                      "break_point": null,
                      "set_point": null,
                      "match_point": null
                  },
                  ................
              ]
          },
          ...........
      ],
      "scores": [
          {
              "score_first": "2",
              "score_second": "5",
              "score_set": "1"
          },
          ............
      ],
      "statistics": []
  },
Javascript call example

var APIkey='!_your_account_APIkey_!';

var socket  = new WebSocket('wss://wss.api-tennis.com/live?APIkey='+APIkey+'&timezone=+03:00');
socket.onmessage = function(e) {
  if (e.data) {
    var matchesData = JSON.parse(e.data);
    // Now variable matchesData contains all matches that received an update
    // Here can update matches in dom from variable matchesData
    console.log(matchesData);
  }
}
                             