import time
from simulator.traffic_light import TrafficLight
from simulator.traffic_controller import TrafficController

def test_adaptive_green():
    lights = [TrafficLight(id="n1", x=0,y=0, orientation="NS"),
              TrafficLight(id="e1", x=1,y=0, orientation="EW")]
    ctrl = TrafficController(lights, min_green=5.0, max_green=30.0, yellow=2.0, ema_alpha=0.5)
    # scenario: heavy NS queue
    density = {"NORTH":{"waiting":10,"moving":0}, "SOUTH":{"waiting":5,"moving":0}, "EAST":{"waiting":0,"moving":0}, "WEST":{"waiting":0,"moving":0}}
    g = ctrl.compute_adaptive_green(density)
    assert g > 5.0
    assert g <= 30.0

def test_emergency_override():
    lights = [TrafficLight(id="n1", x=0,y=0, orientation="NS"),
              TrafficLight(id="e1", x=1,y=0, orientation="EW")]
    ctrl = TrafficController(lights)
    ctrl.emergency_override("EW", duration=6.0)
    assert ctrl.override_active
    assert ctrl.phase == "EW"
