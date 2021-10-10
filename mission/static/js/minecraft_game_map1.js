
var advice = 'Get ready to start game! Follow my navigation instructions to reach as many victims as possible. First, proceed forward.';

var victim_save_dict = {list: {}};
var victim_save_record = {};
var victim_save_dict_index = 0;

var prev_x_level_2 = 0;
var prev_y_level_2 = 0;

var Mrpas = (function () {
    function Mrpas(mapWidth, mapHeight, isTransparent) {
        this.mapWidth = mapWidth;
        this.mapHeight = mapHeight;
        this.isTransparent = isTransparent;
    }
    Mrpas.prototype.computeOctantY = function (deltaX, deltaY, data) {
        var startSlopes = [];
        var endSlopes = [];
        var iteration = 1;
        var totalObstacles = 0;
        var obstaclesInLastLine = 0;
        var minSlope = 0;
        var x;
        var y;
        var halfSlope;
        var processedCell;
        var visible;
        var extended;
        var centreSlope;
        var startSlope;
        var endSlope;
        var previousEndSlope;
        for (y = data.originY + deltaY; y >= data.minY && y <= data.maxY; y += deltaY, obstaclesInLastLine = totalObstacles, ++iteration) {
            halfSlope = 0.5 / iteration;
            previousEndSlope = -1;
            for (processedCell = Math.floor(minSlope * iteration + 0.5), x = data.originX + (processedCell * deltaX); processedCell <= iteration && x >= data.minX && x <= data.maxX; x += deltaX, ++processedCell, previousEndSlope = endSlope) {
                visible = true;
                extended = false;
                centreSlope = processedCell / iteration;
                startSlope = previousEndSlope;
                endSlope = centreSlope + halfSlope;
                if (obstaclesInLastLine > 0) {
                    if (!(data.isVisible(x, y - deltaY) &&
                        this.isTransparent(x, y - deltaY)) &&
                        !(data.isVisible(x - deltaX, y - deltaY) &&
                            this.isTransparent(x - deltaX, y - deltaY))) {
                        visible = false;
                    }
                    else {
                        for (var idx = 0; idx < obstaclesInLastLine && visible; ++idx) {
                            if (startSlope <= endSlopes[idx] && endSlope >= startSlopes[idx]) {
                                if (this.isTransparent(x, y)) {
                                    if (centreSlope > startSlopes[idx] && centreSlope < endSlopes[idx]) {
                                        visible = false;
                                        break;
                                    }
                                }
                                else {
                                    if (startSlope >= startSlopes[idx] && endSlope <= endSlopes[idx]) {
                                        visible = false;
                                        break;
                                    }
                                    else {
                                        startSlopes[idx] = Math.min(startSlopes[idx], startSlope);
                                        endSlopes[idx] = Math.max(endSlopes[idx], endSlope);
                                        extended = true;
                                    }
                                }
                            }
                        }
                    }
                }
                if (visible) {
                    data.setVisible(x, y);
                    if (!this.isTransparent(x, y)) {
                        if (minSlope >= startSlope) {
                            minSlope = endSlope;
                        }
                        else if (!extended) {
                            startSlopes[totalObstacles] = startSlope;
                            endSlopes[totalObstacles++] = endSlope;
                        }
                    }
                }
            }
        }
    };
    Mrpas.prototype.computeOctantX = function (deltaX, deltaY, data) {
        var startSlopes = [];
        var endSlopes = [];
        var iteration = 1;
        var totalObstacles = 0;
        var obstaclesInLastLine = 0;
        var minSlope = 0;
        var x;
        var y;
        var halfSlope;
        var processedCell;
        var visible;
        var extended;
        var centreSlope;
        var startSlope;
        var endSlope;
        var previousEndSlope;
        for (x = data.originX + deltaX; x >= data.minX && x <= data.maxX; x += deltaX, obstaclesInLastLine = totalObstacles, ++iteration) {
            halfSlope = 0.5 / iteration;
            previousEndSlope = -1;
            for (processedCell = Math.floor(minSlope * iteration + 0.5), y = data.originY + (processedCell * deltaY); processedCell <= iteration && y >= data.minY && y <= data.maxY; y += deltaY, ++processedCell, previousEndSlope = endSlope) {
                visible = true;
                extended = false;
                centreSlope = processedCell / iteration;
                startSlope = previousEndSlope;
                endSlope = centreSlope + halfSlope;
                if (obstaclesInLastLine > 0) {
                    if (!(data.isVisible(x - deltaX, y) &&
                        this.isTransparent(x - deltaX, y)) &&
                        !(data.isVisible(x - deltaX, y - deltaY) &&
                            this.isTransparent(x - deltaX, y - deltaY))) {
                        visible = false;
                    }
                    else {
                        for (var idx = 0; idx < obstaclesInLastLine && visible; ++idx) {
                            if (startSlope <= endSlopes[idx] && endSlope >= startSlopes[idx]) {
                                if (this.isTransparent(x, y)) {
                                    if (centreSlope > startSlopes[idx] && centreSlope < endSlopes[idx]) {
                                        visible = false;
                                        break;
                                    }
                                }
                                else {
                                    if (startSlope >= startSlopes[idx] && endSlope <= endSlopes[idx]) {
                                        visible = false;
                                        break;
                                    }
                                    else {
                                        startSlopes[idx] = Math.min(startSlopes[idx], startSlope);
                                        endSlopes[idx] = Math.max(endSlopes[idx], endSlope);
                                        extended = true;
                                    }
                                }
                            }
                        }
                    }
                }
                if (visible) {
                    data.setVisible(x, y);
                    if (!this.isTransparent(x, y)) {
                        if (minSlope >= startSlope) {
                            minSlope = endSlope;
                        }
                        else if (!extended) {
                            startSlopes[totalObstacles] = startSlope;
                            endSlopes[totalObstacles++] = endSlope;
                        }
                    }
                }
            }
        }
    };
    Mrpas.prototype.setMapDimensions = function (mapWidth, mapHeight) {
        this.mapWidth = mapWidth;
        this.mapHeight = mapHeight;
    };
    Mrpas.prototype.compute = function (originX, originY, radius, isVisible, setVisible) {
        setVisible(originX, originY);
        var data = {
            minX: Math.max(0, originX - radius),
            minY: Math.max(0, originY - radius),
            maxX: Math.min(this.mapWidth - 1, originX + radius),
            maxY: Math.min(this.mapHeight - 1, originY + radius),
            originX: originX,
            originY: originY,
            radius: radius,
            isVisible: isVisible,
            setVisible: setVisible
        };
        this.computeOctantY(1, 1, data);
        this.computeOctantX(1, 1, data);
        this.computeOctantY(1, -1, data);
        this.computeOctantX(1, -1, data);
        this.computeOctantY(-1, 1, data);
        this.computeOctantX(-1, 1, data);
        this.computeOctantY(-1, -1, data);
        this.computeOctantX(-1, -1, data);

        // this.computeOctantY(2, 2, data);
        // this.computeOctantX(2, 2, data);
        // this.computeOctantY(2, -2, data);
        // this.computeOctantX(2, -2, data);
        // this.computeOctantY(-2, 2, data);
        // this.computeOctantX(-2, 2, data);
        // this.computeOctantY(-2, -2, data);
        // this.computeOctantX(-2, -2, data);

        // this.computeOctantY(2, 1, data);
        // this.computeOctantX(1, 2, data);
        // this.computeOctantY(1, -2, data);
        // this.computeOctantX(2, -1, data);
        // this.computeOctantY(-2, 1, data);
        // this.computeOctantX(-1, 2, data);
        // this.computeOctantY(-1, -2, data);
        // this.computeOctantX(-2, -1, data);

        // this.computeOctantY(3, 3, data);
        // this.computeOctantX(3, 3, data);
        // this.computeOctantY(3, -3, data);
        // this.computeOctantX(3, -3, data);
        // this.computeOctantY(-3, 3, data);
        // this.computeOctantX(-3, 3, data);
        // this.computeOctantY(-3, -3, data);
        // this.computeOctantX(-3, -3, data);

        // this.computeOctantY(3, 2, data);
        // this.computeOctantX(2, 3, data);
        // this.computeOctantY(2, -3, data);
        // this.computeOctantX(3, -2, data);
        // this.computeOctantY(-3, 2, data);
        // this.computeOctantX(-2, 3, data);
        // this.computeOctantY(-2, -3, data);
        // this.computeOctantX(-3, -2, data);

        // this.computeOctantY(3, 1, data);
        // this.computeOctantX(1, 3, data);
        // this.computeOctantY(1, -3, data);
        // this.computeOctantX(3, -1, data);
        // this.computeOctantY(-3, 1, data);
        // this.computeOctantX(-1, 3, data);
        // this.computeOctantY(-1, -3, data);
        // this.computeOctantX(-3, -1, data);


        // this.computeOctantY(5, 5, data);
        // this.computeOctantX(5, 5, data);
        // this.computeOctantY(5, -5, data);
        // this.computeOctantX(5, -5, data);
        // this.computeOctantY(-5, 5, data);
        // this.computeOctantX(-5, 5, data);
        // this.computeOctantY(-5, -5, data);
        // this.computeOctantX(-5, -5, data);
    };
    return Mrpas;
}());

var score = 0;

var initialTime = 251;
var timedEvent;
var timer_text;
var player;
var player_heading = 2;
var current_room = "In Room: Game Entrance";

var player_x_send = 28;
var player_y_send = 11;
var mturk_key = "";
var offset_tiles = 3;
var display_heading = 'EAST';

var coach_index = document.getElementById("coach_id").value;
var converted_coach_index;

if (round_number == 1){
    if (coach_index == 1){converted_coach_index = 1;}
    if (coach_index == 2){converted_coach_index = 1;}
    if (coach_index == 3){converted_coach_index = 1;}
    if (coach_index == 4){converted_coach_index = 1;}
    if (coach_index == 5){converted_coach_index = 2;}
    if (coach_index == 6){converted_coach_index = 2;}
    if (coach_index == 7){converted_coach_index = 2;}
    if (coach_index == 8){converted_coach_index = 2;}
    if (coach_index == 9){converted_coach_index = 3;}
    if (coach_index == 10){converted_coach_index = 3;}
    if (coach_index == 11){converted_coach_index = 3;}
    if (coach_index == 12){converted_coach_index = 3;}
}
if (round_number == 2){converted_coach_index = 4;}

//var quiz_score = parseInt(document.getElementById("num_correct").value);
//var quiz_score = "1";
//console.log("quiz_score = "+quiz_score);

// CREATE VICTIM SPACE CLICK DICTIONARY
var yellow_victim_space_dict = {
  0: {
      'x': 36,
      'y': 8,
      'num_spaces': 0
      },
  1: {
      'x': 79,
      'y': 22,
      'num_spaces': 0
      },
  2: {
      'x': 46,
      'y': 48,
      'num_spaces': 0
      },
  3: {
      'x': 20,
      'y': 48,
      'num_spaces': 0
      },
  4: {
      'x': 32,
      'y': 32,
      'num_spaces': 0
      },
  5: {
      'x': 49,
      'y': 24,
      'num_spaces': 0
      },
  6: {
      'x': 68,
      'y': 19,
      'num_spaces': 0
      },
  7: {
      'x': 66,
      'y': 31,
      'num_spaces': 0
      },
  };
var green_victim_space_dict = {
  0: {
      'x': 31,
      'y': 5,
      'num_spaces': 0
      },
  1: {
      'x': 51,
      'y': 6,
      'num_spaces': 0
      },
  2: {
      'x': 49,
      'y': 7,
      'num_spaces': 0
      },
  3: {
      'x': 75,
      'y': 7,
      'num_spaces': 0
      },
  4: {
      'x': 88,
      'y': 5,
      'num_spaces': 0
      },
  5: {
      'x': 83,
      'y': 34,
      'num_spaces': 0
      },
  6: {
      'x': 82,
      'y': 49,
      'num_spaces': 0
      },
  7: {
      'x': 64,
      'y': 48,
      'num_spaces': 0
      },
  8: {
      'x': 38,
      'y': 48,
      'num_spaces': 0
      },
  9: {
      'x': 28,
      'y': 44,
      'num_spaces': 0
      },
  10: {
      'x': 23,
      'y': 26,
      'num_spaces': 0
      },
  11: {
      'x': 41,
      'y': 34,
      'num_spaces': 0
      },
  12: {
      'x': 41,
      'y': 21,
      'num_spaces': 0
      },
  13: {
      'x': 49,
      'y': 35,
      'num_spaces': 0
      },
};

// DATABASE SAVING VARIABLES
var db_quiz_score = quiz_score;
var db_condition = condition_number;
var db_uid = name_id;
var db_episode = round_number;
var db_numSteps = 0;
var db_traces = [];
var db_advice = "";
var db_score = 0;
var db_saving_bool = 'false';
var db_victim_pos = '';
var past_trajectory = {'x':[],'y':[]};
var complexity_level = 2;

var turn_left_flag = false;
var turn_right_flag = false;
var get_advice_every_K = 0;
var advice_time_interval = 10;
var advice_steps_interval = 10;

if (db_condition == 1){
    advice_time_interval = 2;
    advice_steps_interval = 0;
}
if (db_condition == 3){
    advice_time_interval = 5;
}
if (complexity_level == 1){
    advice_time_interval = 2;
    advice_steps_interval = 0;
}
if (complexity_level == 4){
    advice_time_interval = 2;
    advice_steps_interval = 0;
}
if (complexity_level == 3){
    advice_time_interval = 5;
}

function isEven(n) {
   return n % 2 == 0;
}
// WE NEED 3 WEBSOCKETS: POSITION, ADVICE, ROOM
//ws.onmessage = function(event) {
//    var messages = document.getElementById('messages')
//    document.getElementById("messages").innerHTML = event.data
//};

//var position_ws = new WebSocket("ws://0.0.0.0:7800/position2");
function send_position_message() {
    var input = JSON.stringify({x:player_x_send, y:player_y_send})
    position_ws.send(input)
}

//var advice_ws = new WebSocket("ws://0.0.0.0:7800/advice2");
function send_advice_message(orient, record) {
    var input = JSON.stringify({x:player_x_send, y:player_y_send, record:record, heading: orient, coach: converted_coach_index,
    past:past_trajectory, level:complexity_level, time:initialTime})
    advice_ws.send(input)
}

//var room_ws = new WebSocket("ws://0.0.0.0:7800/room2");
function send_room_position_message() {
    var input = JSON.stringify({x:player_x_send, y:player_y_send})
    room_ws.send(input)
}

function wait(ms){
   var start = new Date().getTime();
   var end = start;
   while(end < start + ms) {
     end = new Date().getTime();
  }
}



class GameScene extends Phaser.Scene {

    constructor ()
    {
        super('GameScene');
        this.layer;

        this.player_orientation;
        this.x_offset;
        this.y_offset;

        this.yellow_victims;
        this.green_victims;


    }

    preload ()
    {
        this.load.image('tiles', '../static/assets/tilemaps/tiles/attempt_9_resized.png');
        this.load.image('car', '../static/assets/sprites/red_t4_resized.png');
        this.load.tilemapCSV('map', '../static/assets/tilemaps/csv/minecraft_csv_map_2_stairs_free.csv');
        this.load.image('player','../static/assets/sprites/phaser-dude.png');
        this.load.spritesheet('dude', '../static/assets/dude.png', { frameWidth: 32, frameHeight: 48 });
        this.load.image('yellow_victim','../static/assets/sprites/red_ball.png');
        this.load.image('green_victim','../static/assets/sprites/blue_ball.png');

    }

    create ()
    {

        this.physics.world.setFPS(2);
        this.map = this.make.tilemap({ key: 'map', tileWidth: 64, tileHeight: 64 });
        this.tileset = this.map.addTilesetImage('tiles', null, 30, 30, 0, 0);
        this.layer = this.map.createLayer(0, this.tileset, 0, 0);
        this.groundLayer = this.map.createBlankDynamicLayer('Ground', this.tileset)

//        player = this.physics.add.sprite(28*64+32, 11*64+32, 'car');
//        player = this.physics.add.sprite(6*64+32, 5*64+32, 'car');
        // 11,28
        player = this.physics.add.sprite(28*64+32, 11*64+32, 'car');

//        player = this.physics.add.sprite(6*64+32, 5*64+32, 'car');
        player.angle = 0;
        this.player_orientation = 2;
        this.x_offset = 64;
        this.y_offset = 0;

        this.create_victim_groups();


        this.physics.add.overlap(player, this.yellow_victims, this.collectYellow, null, this);
        this.physics.add.overlap(player, this.green_victims, this.collectGreen, null, this);

        this.cursors = this.input.keyboard.createCursorKeys();

        this.physics.world.setBounds(0, 0, 95*64, 52*64);

        this.cameras.main.startFollow(player, true, 0.1, 0.1, 0, 0);
        this.cameras.main.setRotation(-Math.PI/2);

        player.setCollideWorldBounds(true);


        // CREATE MINIMAP
//        this.minimap = this.cameras.add(400, -100, 400, 220).setZoom(0.03).setName('mini');
        this.minimap = this.cameras.add(425, -189, 800, 420).setZoom(0.047).setName('mini');

        this.fov = new Mrpas(this.map.width, this.map.height, (x, y) => {
            const tile = this.layer.getTileAt(x, y)
            return tile && !tile.index==1 && !tile.collides
        });

        this.upFlag = false;
        this.spaceFlag = false;
        this.leftFlag = false;
        this.rightFlag = false;

        this.computeFOV = function ()
        {
            const camera = this.cameras.main;
            var bounds = new Phaser.Geom.Rectangle(
                this.map.worldToTileX(0),
                this.map.worldToTileY(0),
                this.map.worldToTileX(this.physics.world.bounds.width),
                this.map.worldToTileY(this.physics.world.bounds.height),
            )
            for (let y = bounds.y; y < bounds.y + bounds.height; y++)
            {
                for (let x = bounds.x; x < bounds.x + bounds.width; x++)
                {
                    // if (y < 0 || y >= this.map.height || x < 0 || x >= this.map.width)
                    // {
                    //     continue
                    // }

                    const tile = this.layer.getTileAt(x, y)
                    // if (!tile)
                    // {
                    //     continue
                    // }

                    // tile.alpha = 0
                    tile.tint = 0xffffff
                    tile.alpha =  1
                    // tile.visible = true;
                    // tile.tint = 0x404040
                }
            }

            if (!this.fov || !this.map || !this.layer || !player)
            {
                return
            }
            bounds = new Phaser.Geom.Rectangle(
                this.map.worldToTileX(camera.worldView.x) - 1,
                this.map.worldToTileY(camera.worldView.y) - 1,
                this.map.worldToTileX(camera.worldView.width) + 4,
                this.map.worldToTileY(camera.worldView.height) + 4
            )


            // set all tiles within camera view to invisible
            // console.log(bounds);
            for (let y = bounds.y; y < bounds.y + bounds.height; y++)
            {
                for (let x = bounds.x; x < bounds.x + bounds.width; x++)
                {
                    if (y < 0 || y >= this.map.height || x < 0 || x >= this.map.width)
                    {
                        continue
                    }

                    const tile = this.layer.getTileAt(x, y)
                    if (!tile)
                    {
                        // tile.alpha = 1
                        continue
                    }

                    // tile.alpha = 0
                    tile.alpha = 1
//                    tile.tint = 0x404040
                    tile.tint = 0x050505
                }
            }
            this.yellow_victims.getChildren().forEach(function(victim) {
                var x = victim.x;
                var y = victim.y;
                victim.alpha = 0;

            }, this);

            this.green_victims.getChildren().forEach(function(victim) {
                var x = victim.x;
                var y = victim.y;
                victim.alpha = 0;

            }, this);

            // calculate fov here...
            // get player's position
            const px = this.map.worldToTileX(player.x)
            const py = this.map.worldToTileY(player.y)

            // compute fov from player's position
            this.fov.compute(
                px,
                py,
                5,
                (x, y) => {
                    const tile = this.layer.getTileAt(x, y)
                    if (!tile)
                    {
                        return false
                    }

                    return tile.tint === 0xffffff

                },
                (x, y) => {
                    const tile = this.layer.getTileAt(x, y)
                    if (!tile)
                    {
                        return
                    }
                    // tile.alpha = 1
                    const d = Phaser.Math.Distance.Between(py, px, y, x)
                    const alpha = Math.min(2 - d / 6, 1)

                    tile.tint = 0xffffff
                    tile.alpha =  alpha

                    this.yellow_victims.getChildren().forEach(function(victim) {

                        if (victim.x == x*64+32 && victim.y == y*64+32)
                            {
                                victim.alpha =  1;
                            }

                    }, this);
                    this.green_victims.getChildren().forEach(function(victim) {
                        if (victim.x == x*64+32 && victim.y == y*64+32)
                            {
                                victim.alpha =  1;
                            }

                    }, this);
                }
            )
        }

        var initData = {"userid": db_uid, "episode":db_episode, "saving_bool":"", "victim_pos":"",
          "num_step":0, "time_spent":initialTime, "trajectory":"", "advice_message":"", "condition":db_condition,
          "player_score":score, "quiz_score":db_quiz_score, "survey_key":mturk_key};
        writeData(initData);


    }

    create_victim_groups = function() {
        this.yellow_victims = this.physics.add.group({
            key: "yellow_victims",
        });

        this.yellow_victims.create(36*64+32, 8*64+32, 'yellow_victim');
//        this.yellow_victims.create( 83*64+32, 19*64+32,'yellow_victim');
        this.yellow_victims.create(79*64+32, 22*64+32, 'yellow_victim');
        this.yellow_victims.create( 46*64+32,48*64+32, 'yellow_victim');
//        this.yellow_victims.create( 21*64+32, 44*64+32,'yellow_victim');
        this.yellow_victims.create( 20*64+32, 48*64+32, 'yellow_victim');
        this.yellow_victims.create(32*64+32, 32*64+32, 'yellow_victim');
        this.yellow_victims.create( 49*64+32, 24*64+32,'yellow_victim');
        this.yellow_victims.create( 68*64+32, 19*64+32,'yellow_victim');
        this.yellow_victims.create(66*64+32, 31*64+32, 'yellow_victim');

        this.green_victims = this.physics.add.group({
            key: "green_victims",
        });

//        this.green_victims.create(36*64+32, 3*64+32, 'green_victim');
        this.green_victims.create(31*64+32, 5*64+32, 'green_victim');
//        this.green_victims.create( 49*64+32, 5*64+32,'green_victim');
        this.green_victims.create( 51*64+32, 6*64+32,'green_victim');
        this.green_victims.create( 49*64+32, 7*64+32,'green_victim');
        this.green_victims.create( 75*64+32, 7*64+32,'green_victim');
        this.green_victims.create( 88*64+32,5*64+32, 'green_victim');
//        this.green_victims.create( 90*64+32, 14*64+32,'green_victim');
//        this.green_victims.create( 84*64+32, 22*64+32,'green_victim');
//        this.green_victims.create( 84*64+32, 26*64+32,'green_victim');
//        this.green_victims.create( 84*64+32,28*64+32, 'green_victim');
        this.green_victims.create( 83*64+32, 34*64+32,'green_victim');
//        this.green_victims.create( 84*64+32, 43*64+32,'green_victim');
        this.green_victims.create( 82*64+32, 49*64+32,'green_victim');
        this.green_victims.create( 64*64+32, 48*64+32,'green_victim');
        this.green_victims.create( 38*64+32, 48*64+32,'green_victim');
        this.green_victims.create( 28*64+32, 44*64+32,'green_victim');
//        this.green_victims.create( 30*64+32, 49*64+32,'green_victim');
//        this.green_victims.create( 17*64+32, 48*64+32,'green_victim');
        this.green_victims.create( 23*64+32, 26*64+32,'green_victim');
        this.green_victims.create( 41*64+32, 34*64+32,'green_victim');
        this.green_victims.create( 41*64+32,21*64+32, 'green_victim');
//        this.green_victims.create( 46*64+32, 24*64+32,'green_victim');
        this.green_victims.create(49*64+32, 35*64+32, 'green_victim');
    }

    collectYellow = function (player, yellow_victim)
        {
            for (var vic_key in yellow_victim_space_dict) {
                if (yellow_victim_space_dict[vic_key]['x']==this.map.worldToTileX(player.x) &&
                yellow_victim_space_dict[vic_key]['y']==this.map.worldToTileX(player.y)){

                    if (yellow_victim_space_dict[vic_key]['num_spaces']>0){
                        yellow_victim.disableBody(true, true);
                        score += 30;
                        this.events.emit('addScore');
                        const victim_save_add = {
                            color: 'yellow',
                            x: this.map.worldToTileX(player.x)-1,
                            y: this.map.worldToTileY(player.y)-1,
                          };
                        var saveData = {"userid": db_uid, "episode":db_episode, "saving_bool":"saved", "victim_pos":"("+this.map.worldToTileX(player.x)+","+this.map.worldToTileX(player.y)+", yellow)",
                          "num_step":db_numSteps, "time_spent":initialTime, "trajectory":db_traces.join(";"), "advice_message":"", "condition":db_condition,
                          "player_score":score, "quiz_score":db_quiz_score, "survey_key":mturk_key};
                        writeData(saveData);
                        victim_save_record[victim_save_dict_index] = victim_save_add;
                        victim_save_dict_index += 1;
                        victim_save_dict.list = victim_save_record;
                        // GET NEW INSTRUCTION
                        player_x_send = this.map.worldToTileX(player.x);
                        player_y_send = this.map.worldToTileY(player.y);

                        player_heading = this.player_orientation;
                        send_position_message();
                        send_advice_message(this.player_orientation, victim_save_record);
                        send_room_position_message();
                    }
                }
            }
        }
    collectGreen = function (player, green_victim)
        {
            for (var vic_key in green_victim_space_dict) {
                if (green_victim_space_dict[vic_key]['x']==this.map.worldToTileX(player.x) &&
                green_victim_space_dict[vic_key]['y']==this.map.worldToTileX(player.y)){
                    if (green_victim_space_dict[vic_key]['num_spaces']>0){
                        green_victim.disableBody(true, true);
                        score += 10;
                        this.events.emit('addScore');
                        const victim_save_add = {
                            color: 'green',
                            x: this.map.worldToTileX(player.x)-1,
                            y: this.map.worldToTileY(player.y)-1,
                          };
                        var saveData = {"userid": db_uid, "episode":db_episode, "saving_bool":"saved", "victim_pos":"("+this.map.worldToTileX(player.x)+","+this.map.worldToTileX(player.y)+", green)",
                          "num_step":db_numSteps, "time_spent":initialTime, "trajectory":db_traces.join(";"), "advice_message":"", "condition":db_condition,
                          "player_score":score, "quiz_score":db_quiz_score, "survey_key":mturk_key};
                        writeData(saveData);
                        victim_save_record[victim_save_dict_index] = victim_save_add;
                        victim_save_dict_index += 1;
                        victim_save_dict.list = victim_save_record;
                        // GET NEW INSTRUCTION
                        player_x_send = this.map.worldToTileX(player.x);
                        player_y_send = this.map.worldToTileY(player.y);

                        player_heading = this.player_orientation;
                        send_position_message();
                        send_advice_message(this.player_orientation, victim_save_record);
                        send_room_position_message();
                    }
                }
            }
        }


    update ()
    {
//        if (initialTime < 200 && initialTime % 2 == 1){
//            send_advice_message(this.player_orientation, victim_save_record);
//        }
//        if (initialTime == 240){
//            send_advice_message(this.player_orientation, victim_save_record);
//        }


        if (initialTime == 120){
            this.yellow_victims.getChildren().forEach(function(victim) {
                victim.disableBody(true, true);
            }, this);
            send_advice_message(this.player_orientation, victim_save_record);
        }


//        if (initialTime % advice_time_interval == 0 && initialTime < 245){
//            send_advice_message(this.player_orientation, victim_save_record);
//        }

        const cam = this.cameras.main;
        this.computeFOV();

        if (turn_left_flag == true){
            game.paused = true;
            wait(200);

            player.rotation -= Math.PI/4;
            cam.rotation += Math.PI/4;
            if (cam.rotation >= 2*Math.PI){
                cam.rotation = 0;
            }
            if (player.rotation <= -2*Math.PI){
                player.rotation = 0;
            }
            game.paused = false;
            turn_left_flag = false;

            if (player.rotation < 0){
                if (Math.abs(player.rotation - 0) < 0.5){
                    this.player_orientation = 2
                }
                else if (Math.abs(player.rotation + Math.PI/2) < 0.5){
                    this.player_orientation = 1
                }
                else if (Math.abs(player.rotation + Math.PI) < 0.5){
                    this.player_orientation = 4
                }
                else if (Math.abs(player.rotation + Math.PI*3/2) < 0.5){
                    this.player_orientation = 1
                }
            }
            else if (player.rotation >= 0){
                if (Math.abs(player.rotation - 0) < 0.5){
                    this.player_orientation = 2
                }
                else if (Math.abs(player.rotation - Math.PI/2) < 0.5){
                    this.player_orientation = 3
                }
                else if (Math.abs(player.rotation - Math.PI) < 0.5){
                    this.player_orientation = 4
                }
                else if (Math.abs(player.rotation - Math.PI*3/2) < 0.5){
                    this.player_orientation = 1
                }
            }
            player_heading = this.player_orientation;
            send_advice_message(this.player_orientation, victim_save_record);
            this.events.emit('turnedEvent');
            return
        }
        if (turn_right_flag == true){
            game.paused = true;
            wait(200);

            cam.rotation -= Math.PI/4;
            player.rotation += Math.PI/4;
            if (cam.rotation <= -2*Math.PI){
                cam.rotation = 0;
            }
            if (player.rotation >= 2*Math.PI){
                player.rotation = 0;
            }
            game.paused = false;
            turn_right_flag = false;

            if (player.rotation < 0){
                if (Math.abs(player.rotation - 0) < 0.5){
                    this.player_orientation = 2
                }
                else if (Math.abs(player.rotation + Math.PI/2) < 0.5){
                    this.player_orientation = 1
                }
                else if (Math.abs(player.rotation + Math.PI) < 0.5){
                    this.player_orientation = 4
                }
                else if (Math.abs(player.rotation + Math.PI*3/2) < 0.5){
                    this.player_orientation = 1
                }
            }
            else if (player.rotation >= 0){
                if (Math.abs(player.rotation - 0) < 0.5){
                    this.player_orientation = 2
                }
                else if (Math.abs(player.rotation - Math.PI/2) < 0.5){
                    this.player_orientation = 3
                }
                else if (Math.abs(player.rotation - Math.PI) < 0.5){
                    this.player_orientation = 4
                }
                else if (Math.abs(player.rotation - Math.PI*3/2) < 0.5){
                    this.player_orientation = 1
                }
            }
            player_heading = this.player_orientation;
            send_advice_message(this.player_orientation, victim_save_record);
            this.events.emit('turnedEvent');
            return
        }

        if(this.cursors.left.isDown){this.leftFlag=true;}
        if(this.cursors.right.isDown){this.rightFlag=true;}
        if(this.cursors.up.isDown){this.upFlag=true;}
        if(this.cursors.space.isDown){this.spaceFlag=true;}

        if(this.cursors.left.isUp && this.leftFlag==true)
        {
            this.leftFlag=false;
            var tile = this.layer.getTileAtWorldXY(player.x, player.y, true);
            player.rotation -= Math.PI/4;
            cam.rotation += Math.PI/4;
            if (cam.rotation >= 2*Math.PI){
                cam.rotation = 0;
            }
            if (player.rotation <= -2*Math.PI){
                player.rotation = 0;
            }
            turn_left_flag = true;
            var turnData = {"userid": db_uid, "episode":db_episode, "saving_bool":"left_turn",
            "victim_pos":"("+this.map.worldToTileX(player.x)+","+this.map.worldToTileX(player.y)+"," + this.player_orientation + "," +player.rotation + "," + current_room +")",
              "num_step":db_numSteps, "time_spent":initialTime, "trajectory":db_traces.join(";"), "advice_message":advice, "condition":db_condition,
              "player_score":score, "quiz_score":db_quiz_score, "survey_key":mturk_key};
            writeData(turnData);
            send_advice_message(this.player_orientation, victim_save_record);
            player_heading = this.player_orientation;
//            this.events.emit('turnedEvent');

        }
        if(this.cursors.right.isUp && this.rightFlag==true)
        {
            this.rightFlag=false;
            var tile = this.layer.getTileAtWorldXY(player.x, player.y, true);
            cam.rotation -= Math.PI/4;
            player.rotation += Math.PI/4;
            if (cam.rotation <= -2*Math.PI){
                cam.rotation = 0;
            }
            if (player.rotation >= 2*Math.PI){
                player.rotation = 0;
            }
            turn_right_flag = true;
            var turnData = {"userid": db_uid, "episode":db_episode, "saving_bool":"right_turn",
            "victim_pos":"("+this.map.worldToTileX(player.x)+","+this.map.worldToTileX(player.y)+"," + this.player_orientation + "," +player.rotation + "," + current_room +")",
              "num_step":db_numSteps, "time_spent":initialTime, "trajectory":db_traces.join(";"), "advice_message":advice, "condition":db_condition,
              "player_score":score, "quiz_score":db_quiz_score, "survey_key":mturk_key};
            writeData(turnData);
            send_advice_message(this.player_orientation, victim_save_record);
            player_heading = this.player_orientation;
//            this.events.emit('turnedEvent');

        }
        if(this.cursors.up.isUp && this.upFlag==true)
        {
            this.upFlag=false;
//            if (player.rotation < 0){
//                if (Math.abs(player.rotation - 0) < 0.5){
//                    this.player_orientation = 2
//                }
//                else if (Math.abs(player.rotation + Math.PI/2) < 0.5){
//                    this.player_orientation = 1
//                }
//                else if (Math.abs(player.rotation + Math.PI) < 0.5){
//                    this.player_orientation = 4
//                }
//                else if (Math.abs(player.rotation + Math.PI*3/2) < 0.5){
//                    this.player_orientation = 1
//                }
//            }
//            else if (player.rotation >= 0){
//                if (Math.abs(player.rotation - 0) < 0.5){
//                    this.player_orientation = 2
//                }
//                else if (Math.abs(player.rotation - Math.PI/2) < 0.5){
//                    this.player_orientation = 3
//                }
//                else if (Math.abs(player.rotation - Math.PI) < 0.5){
//                    this.player_orientation = 4
//                }
//                else if (Math.abs(player.rotation - Math.PI*3/2) < 0.5){
//                    this.player_orientation = 1
//                }
//            }
//            player_heading = this.player_orientation;
            if (this.player_orientation == 1){
                this.x_offset = 0;
                this.y_offset = -64;
                player.angle = -90;

                this.cameras.main.setRotation(0);
            }
            else if (this.player_orientation == 2){
                this.x_offset = 64;
                this.y_offset = 0;
                player.angle = 0;

                this.cameras.main.setRotation(Math.PI*3/2);
            }
            else if (this.player_orientation == 3){
                this.x_offset = 0;
                this.y_offset = 64;
                player.angle = 90;

                this.cameras.main.setRotation(Math.PI);
            }
            else if (this.player_orientation == 4){
                this.x_offset = -64;
                this.y_offset = 0;
                player.angle = 180;

                this.cameras.main.setRotation(Math.PI/2);
            }
            var tile = this.layer.getTileAtWorldXY(player.x + this.x_offset, player.y + this.y_offset, true);

            if (tile.index == 1)
            {
                //  Blocked, we can't move
            }
            else
            {
                player.x += this.x_offset;
                player.y += this.y_offset;
                past_trajectory['x'].push("("+this.map.worldToTileX(player.x)+","+this.map.worldToTileX(player.y)+")");

                db_numSteps += 1;
                // player.angle = -90;
                // player.anims.play("turn", true);
//                var position_emission = {x: this.map.worldToTileX(player.x), y: this.map.worldToTileY(player.y)};
                player_x_send = this.map.worldToTileX(player.x);
                player_y_send = this.map.worldToTileY(player.y);

                player_heading = this.player_orientation;
                send_position_message();

                send_advice_message(this.player_orientation, victim_save_record);

//                get_advice_every_K += 1;
////                if (get_advice_every_K > 10 || initialTime % 10 == 0){
//                if (get_advice_every_K > advice_steps_interval){
//                    send_advice_message(this.player_orientation, victim_save_record);
//                    get_advice_every_K = 0;
//                }

//                if (Math.abs(player_y_send-prev_y_level_2) + Math.abs(player_x_send - prev_x_level_2) > 5){
//                    send_advice_message(this.player_orientation, victim_save_record);
//                    prev_y_level_2 = player_y_send;
//                    prev_x_level_2 = player_x_send;
//                }

                send_room_position_message();

                var moveData = {"userid": db_uid, "episode":db_episode, "saving_bool":"step_forward",
                "victim_pos":"("+this.map.worldToTileX(player.x)+","+this.map.worldToTileX(player.y)+"," + this.player_orientation + "," + current_room +")",
                  "num_step":db_numSteps, "time_spent":initialTime, "trajectory":db_traces.join(";"), "advice_message":advice, "condition":db_condition,
                  "player_score":score, "quiz_score":db_quiz_score, "survey_key":mturk_key};
                writeData(moveData);
                }
            }
        if(this.cursors.space.isUp && this.spaceFlag==true)
        {
            this.spaceFlag=false;
            for (var vic_key in green_victim_space_dict) {
                if (green_victim_space_dict[vic_key]['x']==this.map.worldToTileX(player.x) &&
                green_victim_space_dict[vic_key]['y']==this.map.worldToTileX(player.y)){
                    green_victim_space_dict[vic_key]['num_spaces'] += 2;
                    var saveData = {"userid": db_uid, "episode":db_episode, "saving_bool":"in_progress",
                    "victim_pos":"("+this.map.worldToTileX(player.x)+","+this.map.worldToTileX(player.y)+", green" +","+current_room+")",
                      "num_step":db_numSteps, "time_spent":initialTime, "trajectory":db_traces.join(";"), "advice_message":advice, "condition":db_condition,
                      "player_score":score, "quiz_score":db_quiz_score, "survey_key":mturk_key};
                    writeData(saveData);
                }
            }
            for (var vic_key in yellow_victim_space_dict) {
                if (yellow_victim_space_dict[vic_key]['x']==this.map.worldToTileX(player.x) &&
                yellow_victim_space_dict[vic_key]['y']==this.map.worldToTileX(player.y)){
                    yellow_victim_space_dict[vic_key]['num_spaces'] += 2;
                    var saveData = {"userid": db_uid, "episode":db_episode, "saving_bool":"in_progress",
                    "victim_pos":"("+this.map.worldToTileX(player.x)+","+this.map.worldToTileX(player.y)+", yellow" +","+current_room+")",
                      "num_step":db_numSteps, "time_spent":initialTime, "trajectory":db_traces.join(";"), "advice_message":advice, "condition":db_condition,
                      "player_score":score, "quiz_score":db_quiz_score, "survey_key":mturk_key};
                    writeData(saveData);
                }
            }
            send_advice_message(this.player_orientation, victim_save_record);

        }

        if (player.rotation < 0){
            if (Math.abs(player.rotation - 0) < 0.5){
                this.player_orientation = 2
            }
            else if (Math.abs(player.rotation + Math.PI/2) < 0.5){
                this.player_orientation = 1
            }
            else if (Math.abs(player.rotation + Math.PI) < 0.5){
                this.player_orientation = 4
            }
            else if (Math.abs(player.rotation + Math.PI*3/2) < 0.5){
                this.player_orientation = 1
            }
        }
        else if (player.rotation >= 0){
            if (Math.abs(player.rotation - 0) < 0.5){
                this.player_orientation = 2
            }
            else if (Math.abs(player.rotation - Math.PI/2) < 0.5){
                this.player_orientation = 3
            }
            else if (Math.abs(player.rotation - Math.PI) < 0.5){
                this.player_orientation = 4
            }
            else if (Math.abs(player.rotation - Math.PI*3/2) < 0.5){
                this.player_orientation = 1
            }
        }
        player_heading = this.player_orientation;
        db_traces.push("("+this.map.worldToTileX(player.x)+","+this.map.worldToTileX(player.y)+", "+this.player_orientation+")");
//        this.events.emit('turnedEvent');
//        past_trajectory['x'].push("("+this.map.worldToTileX(player.x)+","+this.map.worldToTileX(player.y)")");
//        past_trajectory['x'].push(this.map.worldToTileX(player.x));
//        past_trajectory['y'].push(this.map.worldToTileX(player.y));
    }

}


var minimap_image;
var minimap_circle;
var graphics;

class ScoreScene extends Phaser.Scene {

    constructor ()
    {
        super({ key: 'ScoreScene', active: true });

        // this.score = 0;
    }
    preload(){
        this.load.image('emptymap', '../static/assets/emptymap.png');
    }

    create ()
    {

        graphics = this.add.graphics();
        var background_rect = this.add.rectangle(0, 0, 1590, 380, 0x000000);


//        graphics.lineStyle(4, 0xC0C0C0, 1);
//        graphics.strokeRoundedRect(595, 5, 180, 100, { tl: 0, tr: 0, bl: 0, br: 0 });
        graphics.lineStyle(20, 0x81d4fa, 1);
        graphics.strokeRoundedRect(800, 5, 295, 170, { tl: 0, tr: 0, bl: 0, br: 0 });

        //  Our Text object to display the Score
        let info = this.add.text(50, 30, 'Score: 0', { font: '22px Arial', fill: '#ffffff' });
        let advice_text = this.add.text(50, 80, advice, { font: '28px Arial', fill: '#FFFF00' ,
                wordWrap: { width: 700, useAdvancedWrap: true }});

//        let curr_room_display = this.add.text(600, 130, current_room, { font: '24px Arial', fill: '#81d4fa', wordWrap: { width: 150, useAdvancedWrap: true }},);
        let curr_room_display = this.add.text(840, 200, current_room, { font: '24px Arial', fill: '#81d4fa', wordWrap: { width: 250, useAdvancedWrap: true }},);

        let curr_heading_display = this.add.text(840, 300, "Heading: EAST", { font: '24px Arial', fill: '#81d4fa', wordWrap: { width: 250, useAdvancedWrap: true }},);

        timer_text = this.add.text(350, 30, 'GAME BEGINS IN : 10',{ font: '24px Arial', fill: '#ffffff' });

        timedEvent = this.time.addEvent({ delay: 1000, callback: this.onEvent, callbackScope: this, loop: true });

        //  Grab a reference to the Game Scene
        let ourGame = this.scene.get('GameScene');

        //  Listen for events from it
        ourGame.events.on('addScore', function () {

            // this.score += 10;

            info.setText('Score: ' + score);
            document.getElementById("game_score").value = score;

        }, this);

        ourGame.events.on('turnedEvent', function () {

            // UPDATE HEADING
            display_heading = 'EAST';
            if (player_heading == 1){
                display_heading = 'NORTH';
            }
            if (player_heading == 2){
                display_heading = 'EAST';
            }
            if (player_heading == 3){
                display_heading = 'SOUTH';
            }
            if (player_heading == 4){
                display_heading = 'WEST';
            }
            curr_heading_display.setText("Heading: "+display_heading);

        }, this);

        advice_ws.onmessage = function(event) {
//            var advice_msg_text = document.getElementById('advice_text')

            advice_text.setText(""+event.data);
            advice = event.data;
            if ((initialTime <= 120) && (initialTime >= 115)){
                advice_text.setText("2 Minutes up, RED victims all expired. Recalculating path."+event.data);
            }
//            complexity_level = event.data.level;


        };

        room_ws.onmessage = function(event) {
//            var room_name_text = document.getElementById('room_name_text')
            if (event.data != 'undefined'){
                curr_room_display.setText("In Room: "+event.data);
                current_room = event.data;
            }


        };

    }

//    update(){
//        display_heading = 'EAST';
//        if (player_orientation == 1){
//            display_heading = 'NORTH';
//        }
//        if (player_orientation == 2){
//            display_heading = 'EAST';
//        }
//        if (player_orientation == 3){
//            display_heading = 'SOUTH';
//        }
//        if (player_orientation == 4){
//            display_heading = 'WEST';
//        }
//        curr_heading_display.setText("Heading: "+display_heading);
//    }

    formatTime = function (seconds){
        // Minutes
        var minutes = Math.floor(seconds/60);
        // Seconds
        var partInSeconds = seconds%60;
        // Adds left zeros to seconds
        partInSeconds = partInSeconds.toString().padStart(2,'0');
        // Returns formated time
        if (minutes <= 0 && partInSeconds <= 0){
            minutes = 0;
            partInSeconds = 0;
            advice = "GAME OVER!!";
            advice_text.setText(advice);
            var overData = {"userid": db_uid, "episode":db_episode, "saving_bool":"game_over",
            "victim_pos":"("+this.map.worldToTileX(player.x)+","+this.map.worldToTileX(player.y)+"," + this.player_orientation + "," +player.rotation + "," + current_room +")",
              "num_step":db_numSteps, "time_spent":initialTime, "trajectory":db_traces.join(";"), "advice_message":advice, "condition":db_condition,
              "player_score":score, "quiz_score":db_quiz_score, "survey_key":mturk_key};
            writeData(overData);
            //window.location.href = '/post_game_1?round='+round_number +'&condition='+condition_number + '&workerId='+name_id;
        }

        return `${minutes}:${partInSeconds}`;
    }

    onEvent = function ()
    {
        initialTime -= 1; // One second
        timer_text.setText('Countdown: ' + this.formatTime(initialTime));
        if (initialTime == 250){
            timer_text.setText('GAME BEGINS IN : 10');
        }
        if (initialTime == 249){
            timer_text.setText('GAME BEGINS IN : 9');
        }
        if (initialTime == 248){
            timer_text.setText('GAME BEGINS IN : 8');
        }
        if (initialTime == 247){
            timer_text.setText('GAME BEGINS IN : 7');
        }
        if (initialTime == 246){
            timer_text.setText('GAME BEGINS IN : 6');
        }
        if (initialTime == 245){
            timer_text.setText('GAME BEGINS IN : 5');
        }
        if (initialTime == 244){
            timer_text.setText('GAME BEGINS IN : 4');
        }
        if (initialTime == 243){
            timer_text.setText('GAME BEGINS IN : 3');
        }
        if (initialTime == 242){
            timer_text.setText('GAME BEGINS IN : 2');
        }
        if (initialTime == 241){
            timer_text.setText('GAME BEGINS IN : 1');
        }
    }
}


var config = {
    type: Phaser.WEBGL,
//    width: 800,
//    height: 800,
    width: 1100,
    height: 1000,
    parent: 'phaser-example',
    pixelArt: true,
    backgroundColor: '#000000',
    physics: {
        default: 'arcade',
    },
    scene: [ GameScene, ScoreScene ]
};

var game = new Phaser.Game(config);




function writeData(data){
  const dataOptions = {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json'
    },
    body: JSON.stringify(data)
  };
  fetch('/game_play', dataOptions);
}

