
var advice = 'Start game! Find and rescue all red and blue victims.';

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

var initialTime = 63;
var timedEvent;
var timer_text;
var player;
var player_heading = 2;
var current_room = "";

var player_x_send = 6;
var player_y_send = 5;
var mturk_key = "";

var coach_index = document.getElementById("coach_id").value;
//var quiz_score = parseInt(document.getElementById("num_correct").value);
//var quiz_score = "1";
//console.log("quiz_score = "+quiz_score);

// CREATE VICTIM SPACE CLICK DICTIONARY
var yellow_victim_space_dict = {
  0: {
      'x':11,
      'y': 11,
      'num_spaces': 0
      },
  1: {
      'x': 20,
      'y': 22,
      'num_spaces': 0
      },
  2: {
      'x': 20,
      'y': 14,
      'num_spaces': 0
      },
  3: {
      'x': 42,
      'y': 15,
      'num_spaces': 0
      },
  }
var green_victim_space_dict = {
  0: {
      'x': 8,
      'y': 14,
      'num_spaces': 0
      },
  1: {
      'x': 4,
      'y': 14,
      'num_spaces': 0
      },
  2: {
      'x': 20,
      'y': 4,
      'num_spaces': 0
      },
  3: {
      'x': 42,
      'y': 12,
      'num_spaces': 0
      },
  4: {
      'x': 49,
      'y': 5,
      'num_spaces': 0
      },
  5: {
      'x': 49,
      'y': 20,
      'num_spaces': 0
      },
};

// DATABASE SAVING VARIABLES
var db_quiz_score = 0;
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
if (complexity_level == 3){
    advice_time_interval = 5;
}


// WE NEED 3 WEBSOCKETS: POSITION, ADVICE, ROOM
//ws.onmessage = function(event) {
//    var messages = document.getElementById('messages')
//    document.getElementById("messages").innerHTML = event.data
//};

var position_ws = new WebSocket("ws://52.90.55.207:5000/position");
function send_position_message() {
    var input = JSON.stringify({x:player_x_send, y:player_y_send})
    position_ws.send(input)
}

var advice_ws = new WebSocket("ws://52.90.55.207:5000/advice");
function send_advice_message(orient, record) {
//    var input = JSON.stringify({x:player_x_send, y:player_y_send, record:record, heading: orient, coach: coach_index,
//    past:past_trajectory, level:complexity_level, time:initialTime})
//    advice_ws.send(input)
}

var room_ws = new WebSocket("ws://52.90.55.207:5000/room");
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
        this.load.tilemapCSV('map', '../static/assets/tilemaps/csv/minecraft_tilemap_4_practice_map_w_border.csv');
        this.load.image('player','../static/assets/sprites/phaser-dude.png');
        this.load.spritesheet('dude', '../static/assets/dude.png', { frameWidth: 32, frameHeight: 48 });
        this.load.image('yellow_victim','../static/assets/sprites/red_ball.png');
        this.load.image('green_victim','../static/assets/sprites/blue_ball.png');

    }

    create ()
    {

        this.map = this.make.tilemap({ key: 'map', tileWidth: 64, tileHeight: 64 });
        this.tileset = this.map.addTilesetImage('tiles', null, 30, 30, 0, 0);
        this.layer = this.map.createLayer(0, this.tileset, 0, 0);
        this.groundLayer = this.map.createBlankDynamicLayer('Ground', this.tileset)

        player = this.physics.add.sprite(6*64+32, 5*64+32, 'car');
        player.angle = 0;
        this.player_orientation = 2;
        this.x_offset = 64;
        this.y_offset = 0;

        this.create_victim_groups();


        this.physics.add.overlap(player, this.yellow_victims, this.collectYellow, null, this);
        this.physics.add.overlap(player, this.green_victims, this.collectGreen, null, this);

        this.cursors = this.input.keyboard.createCursorKeys();

        this.physics.world.setBounds(0, 0, 23*64, 52*64);

        this.cameras.main.startFollow(player, true, 0.1, 0.1, 0, 0);
        this.cameras.main.setRotation(-Math.PI/2);

        player.setCollideWorldBounds(true);


        // CREATE MINIMAP
        this.minimap = this.cameras.add(480+30, -180, 400, 400).setZoom(0.06).setName('mini');
//        this.minimap = this.cameras.add(225, -189, 800, 420).setZoom(0.047).setName('mini');

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

//        var initData = {"userid": db_uid, "episode":db_episode, "saving_bool":"", "victim_pos":"",
//          "num_step":0, "time_spent":initialTime, "trajectory":"", "advice_message":"", "condition":db_condition,
//          "player_score":score, "quiz_score":db_quiz_score, "survey_key":mturk_key};
//        writeData(initData);


    }

    create_victim_groups = function() {
        this.yellow_victims = this.physics.add.group({
            key: "yellow_victims",
        });

        this.yellow_victims.create(11*64+32, 11*64+32, 'yellow_victim');
        this.yellow_victims.create(20*64+32, 22*64+32, 'yellow_victim');
        this.yellow_victims.create( 20*64+32,14*64+32, 'yellow_victim');
        this.yellow_victims.create( 42*64+32, 15*64+32, 'yellow_victim');

        this.green_victims = this.physics.add.group({
            key: "green_victims",
        });

        this.green_victims.create(8*64+32, 14*64+32, 'green_victim');
        this.green_victims.create(4*64+32, 14*64+32, 'green_victim');
        this.green_victims.create( 20*64+32,4*64+32, 'green_victim');
        this.green_victims.create( 42*64+32, 12*64+32, 'green_victim');
        this.green_victims.create( 49*64+32, 5*64+32, 'green_victim');
        this.green_victims.create( 49*64+32, 20*64+32, 'green_victim');
    }

    collectYellow = function (player, yellow_victim)
        {
            for (var vic_key in yellow_victim_space_dict) {
                if (yellow_victim_space_dict[vic_key]['x']==this.map.worldToTileX(player.x) &&
                yellow_victim_space_dict[vic_key]['y']==this.map.worldToTileX(player.y)){
                    if (yellow_victim_space_dict[vic_key]['num_spaces']>=1){
                        yellow_victim.disableBody(true, true);
                        score += 30;
                        this.events.emit('addScore');
                        const victim_save_add = {
                            color: 'yellow',
                            x: this.map.worldToTileX(player.x)-1,
                            y: this.map.worldToTileY(player.y)-1,
                          };
//                        var saveData = {"userid": db_uid, "episode":db_episode, "saving_bool":"saved", "victim_pos":"("+this.map.worldToTileX(player.x)+","+this.map.worldToTileX(player.y)+", yellow)",
//                          "num_step":db_numSteps, "time_spent":initialTime, "trajectory":db_traces.join(";"), "advice_message":"", "condition":db_condition,
//                          "player_score":score, "quiz_score":db_quiz_score, "survey_key":mturk_key};
//                        writeData(saveData);
                        victim_save_record[victim_save_dict_index] = victim_save_add;
                        victim_save_dict_index += 1;
                        victim_save_dict.list = victim_save_record;
                        // GET NEW INSTRUCTION
                        player_x_send = this.map.worldToTileX(player.x);
                        player_y_send = this.map.worldToTileY(player.y);

                        player_heading = this.player_orientation;
                        send_position_message();
//                        send_advice_message(this.player_orientation, victim_save_record);
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
                    if (green_victim_space_dict[vic_key]['num_spaces']>=1){
                        green_victim.disableBody(true, true);
                        score += 10;
                        this.events.emit('addScore');
                        const victim_save_add = {
                            color: 'green',
                            x: this.map.worldToTileX(player.x)-1,
                            y: this.map.worldToTileY(player.y)-1,
                          };
//                        var saveData = {"userid": db_uid, "episode":db_episode, "saving_bool":"saved", "victim_pos":"("+this.map.worldToTileX(player.x)+","+this.map.worldToTileX(player.y)+", green)",
//                          "num_step":db_numSteps, "time_spent":initialTime, "trajectory":db_traces.join(";"), "advice_message":"", "condition":db_condition,
//                          "player_score":score, "quiz_score":db_quiz_score, "survey_key":mturk_key};
//                        writeData(saveData);
                        victim_save_record[victim_save_dict_index] = victim_save_add;
                        victim_save_dict_index += 1;
                        victim_save_dict.list = victim_save_record;
                        // GET NEW INSTRUCTION
                        player_x_send = this.map.worldToTileX(player.x);
                        player_y_send = this.map.worldToTileY(player.y);

                        player_heading = this.player_orientation;
                        send_position_message();
//                        send_advice_message(this.player_orientation, victim_save_record);
                        send_room_position_message();
                    }
                }
            }
        }


    update ()
    {
//        if (initialTime == 355){
//            send_advice_message(this.player_orientation, victim_save_record);
//        }


        if (initialTime == 120){
            this.yellow_victims.getChildren().forEach(function(victim) {
                victim.disableBody(true, true);
            }, this);
//            send_advice_message(this.player_orientation, victim_save_record);
        }


//        if (initialTime % advice_time_interval == 0 && initialTime < 240){
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
//            var turnData = {"userid": db_uid, "episode":db_episode, "saving_bool":"left_turn",
//            "victim_pos":"("+this.map.worldToTileX(player.x)+","+this.map.worldToTileX(player.y)+"," + this.player_orientation + "," +player.rotation + "," + current_room +")",
//              "num_step":db_numSteps, "time_spent":initialTime, "trajectory":db_traces.join(";"), "advice_message":advice, "condition":db_condition,
//              "player_score":score, "quiz_score":db_quiz_score, "survey_key":mturk_key};
//            writeData(turnData);

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
//            var turnData = {"userid": db_uid, "episode":db_episode, "saving_bool":"right_turn",
//            "victim_pos":"("+this.map.worldToTileX(player.x)+","+this.map.worldToTileX(player.y)+"," + this.player_orientation + "," +player.rotation + "," + current_room +")",
//              "num_step":db_numSteps, "time_spent":initialTime, "trajectory":db_traces.join(";"), "advice_message":advice, "condition":db_condition,
//              "player_score":score, "quiz_score":db_quiz_score, "survey_key":mturk_key};
//            writeData(turnData);

        }
        if(this.cursors.up.isUp && this.upFlag==true)
        {
            this.upFlag=false;
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

                get_advice_every_K += 1;
//                if (get_advice_every_K > 10 || initialTime % 10 == 0){
//                if (get_advice_every_K > advice_steps_interval){
////                    send_advice_message(this.player_orientation, victim_save_record);
//                    get_advice_every_K = 0;
//                }

//                if (Math.abs(player_y_send-prev_y_level_2) + Math.abs(player_x_send - prev_x_level_2) > 5){
//                    send_advice_message(this.player_orientation, victim_save_record);
//                    prev_y_level_2 = player_y_send;
//                    prev_x_level_2 = player_x_send;
//                }

                send_room_position_message();

//                var moveData = {"userid": db_uid, "episode":db_episode, "saving_bool":"step_forward",
//                "victim_pos":"("+this.map.worldToTileX(player.x)+","+this.map.worldToTileX(player.y)+"," + this.player_orientation + "," + current_room +")",
//                  "num_step":db_numSteps, "time_spent":initialTime, "trajectory":db_traces.join(";"), "advice_message":advice, "condition":db_condition,
//                  "player_score":score, "quiz_score":db_quiz_score, "survey_key":mturk_key};
//                writeData(moveData);
                }
            }
        if(this.cursors.space.isUp && this.spaceFlag==true)
        {
            this.spaceFlag=false;
            for (var vic_key in green_victim_space_dict) {
                if (green_victim_space_dict[vic_key]['x']==this.map.worldToTileX(player.x) &&
                green_victim_space_dict[vic_key]['y']==this.map.worldToTileX(player.y)){
                    green_victim_space_dict[vic_key]['num_spaces'] += 2;
//                    var saveData = {"userid": db_uid, "episode":db_episode, "saving_bool":"in_progress",
//                    "victim_pos":"("+this.map.worldToTileX(player.x)+","+this.map.worldToTileX(player.y)+", green" +","+current_room+")",
//                      "num_step":db_numSteps, "time_spent":initialTime, "trajectory":db_traces.join(";"), "advice_message":advice, "condition":db_condition,
//                      "player_score":score, "quiz_score":db_quiz_score, "survey_key":mturk_key};
//                    writeData(saveData);
                }
            }
            for (var vic_key in yellow_victim_space_dict) {
                if (yellow_victim_space_dict[vic_key]['x']==this.map.worldToTileX(player.x) &&
                yellow_victim_space_dict[vic_key]['y']==this.map.worldToTileX(player.y)){
                    yellow_victim_space_dict[vic_key]['num_spaces'] += 2;
//                    var saveData = {"userid": db_uid, "episode":db_episode, "saving_bool":"in_progress",
//                    "victim_pos":"("+this.map.worldToTileX(player.x)+","+this.map.worldToTileX(player.y)+", yellow" +","+current_room+")",
//                      "num_step":db_numSteps, "time_spent":initialTime, "trajectory":db_traces.join(";"), "advice_message":advice, "condition":db_condition,
//                      "player_score":score, "quiz_score":db_quiz_score, "survey_key":mturk_key};
//                    writeData(saveData);
                }
            }
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
        var background_rect = this.add.rectangle(0, 0, 1375, 380, 0x000000);
//        graphics.lineStyle(4, 0xC0C0C0, 1);
//        graphics.strokeRoundedRect(665, 5, 95, 205, { tl: 0, tr: 0, bl: 0, br: 0 });
        graphics.lineStyle(15, 0x81d4fa, 1);
        graphics.strokeRoundedRect(665+30, 5, 95, 205, { tl: 0, tr: 0, bl: 0, br: 0 });

        //  Our Text object to display the Score
        let info = this.add.text(50, 50, 'Score: 0', { font: '22px Arial', fill: '#ffffff' });
        let advice_text = this.add.text(50, 100, advice, { font: '28px Arial', fill: '#FFFF00' ,
                wordWrap: { width: 500, useAdvancedWrap: true }});

//        let curr_room_display = this.add.text(650, 230, 'Current: Practice Room', { font: '24px Arial', fill: '#81d4fa', wordWrap: { width: 150, useAdvancedWrap: true }},);
        let curr_room_display = this.add.text(670, 230, "In Room: Practice Room", { font: '24px Arial', fill: '#81d4fa', wordWrap: { width: 150, useAdvancedWrap: true }},);



        timer_text = this.add.text(250, 50, 'PRACTICE BEGINS IN : 3',{ font: '24px Arial', fill: '#ffffff' });

        timedEvent = this.time.addEvent({ delay: 1000, callback: this.onEvent, callbackScope: this, loop: true });

        //  Grab a reference to the Game Scene
        let ourGame = this.scene.get('GameScene');

        //  Listen for events from it
        ourGame.events.on('addScore', function () {

            // this.score += 10;

            info.setText('Score: ' + score);

        }, this);
//        advice_ws.onmessage = function(event) {
////            var advice_msg_text = document.getElementById('advice_text')
//
//            advice_text.setText("Instructor: "+event.data);
//            advice = event.data;
////            complexity_level = event.data.level;
//        };

        room_ws.onmessage = function(event) {
//            var room_name_text = document.getElementById('room_name_text')
            if (event.data != 'undefined'){
                curr_room_display.setText("In Room: "+event.data);
                current_room = event.data;
            }
        };

    }


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
//            var overData = {"userid": db_uid, "episode":db_episode, "saving_bool":"game_over",
//            "victim_pos":"("+this.map.worldToTileX(player.x)+","+this.map.worldToTileX(player.y)+"," + this.player_orientation + "," +player.rotation + "," + current_room +")",
//              "num_step":db_numSteps, "time_spent":initialTime, "trajectory":db_traces.join(";"), "advice_message":advice, "condition":db_condition,
//              "player_score":score, "quiz_score":db_quiz_score, "survey_key":mturk_key};
//            writeData(overData);
            //window.location.href = '/post_game_1?round='+round_number +'&condition='+condition_number + '&workerId='+name_id;
        }

        return `${minutes}:${partInSeconds}`;
    }

    onEvent = function ()
    {
        initialTime -= 1; // One second
        timer_text.setText('Countdown: ' + this.formatTime(initialTime));
        if (initialTime == 63){
            timer_text.setText('PRACTICE BEGINS IN : 3');
        }
        if (initialTime == 62){
            timer_text.setText('PRACTICE BEGINS IN : 2');
        }
        if (initialTime == 61){
            timer_text.setText('PRACTICE BEGINS IN : 1');
        }
    }
}


var config = {
    type: Phaser.WEBGL,
    width: 800,
    height: 780,
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

