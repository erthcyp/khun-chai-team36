# Prompt Lexicon

## Purpose

This lexicon defines seed terms for each traffic event class. These terms are used by large language models (GPT, Claude, Gemini) to generate 2,000 diverse prompts — 500 per class. The prompts are then filtered by blind voting and used to train a BERT-based prompt classifier.

## Traffic Event Classes

There are four classes in this system:

- Accident — scenes involving road collisions, vehicle damage, and emergency responses
- Normal Flow — normal, unobstructed traffic with low to moderate vehicle density
- Congestion — high vehicle density, queuing, and reduced or stopped movement
- Invalid Scene — non-informative or technically degraded frames that cannot be classified reliably

## Class 1: Accident

Accident scenes involve visible road collisions, damaged vehicles, crash evidence, blocked lanes caused by crashes, or emergency response activity.

### Accident — Seed Terms by Category

#### Direct Event

collision, crash, accident, traffic accident, car crash, road crash, vehicle collision, pile-up, head-on collision, rear-end collision, side-impact crash, rollover accident, multi-vehicle accident

#### Vehicle Damage

damaged vehicle, overturned car, wrecked car, smashed vehicle, crushed car, flipped vehicle, destroyed car, mangled vehicle, disabled vehicle after crash, vehicle with crushed front bumper, vehicle with broken rear bumper

#### Accident Evidence

crash debris, broken glass on road, detached bumper, scattered vehicle parts, skid marks, smoke from vehicle, deployed airbags, oil spill after collision, damaged roadside barrier, vehicle parts on lane

#### Emergency Response

ambulance, police car near crash, fire truck, first responders, rescue team, emergency vehicles on scene, paramedics, road rescue operation, police directing traffic around crash

#### Traffic Impact

blocked lane caused by crash, lane closure due to collision, traffic disruption near damaged vehicles, road hazard from crash debris, emergency detour around accident scene, vehicles stopped around visible crash, stopped traffic behind collision scene, hazard lights near damaged vehicles

#### Road Signs and Safety Equipment

accident ahead sign, road closure sign, emergency cones, warning triangle, flare on road, emergency barrier, caution tape, temporary crash barrier

## Class 2: Normal Flow

Normal Flow scenes show vehicles moving normally on clear and unobstructed roads, with low to moderate vehicle density and no visible traffic jam or incident.

### Normal Flow — Seed Terms by Category

#### Low Density

light traffic, sparse vehicles, few cars on road, low vehicle density, open road, nearly empty road, widely spaced vehicles, light road activity

#### Vehicle Spacing

large gaps between vehicles, vehicles evenly spaced, cars separated by safe distance, no queue visible, enough space between cars, open lanes ahead, clear distance between vehicles

#### Motion

moving smoothly, flowing traffic, vehicles in motion, steady traffic, continuous flow, cars travelling freely, uninterrupted traffic, vehicles maintaining normal speed, cars moving at regular pace

#### Road Condition

clear road, open highway, unobstructed road, well-maintained road, dry road surface, clean road, visible road markings, clear lanes, normal road layout

#### Flow State

free-flow traffic, normal traffic, regular traffic, standard road conditions, normal road operation, orderly traffic, traffic at normal speed, smooth road movement

#### Time Context

morning commute normal flow, off-peak traffic, daytime light traffic, regular road activity, non-rush-hour road, normal weekday traffic

#### No-Congestion Cue

no stop-and-go movement, no long queue, no bumper-to-bumper traffic, no repeated braking, no blocked lane, no visible traffic jam

## Class 3: Congestion

Congestion scenes show high vehicle density, closely packed vehicles, traffic queues, slow movement, stop-and-go traffic, or traffic jam conditions.

### Congestion — Seed Terms by Category

#### Density

dense traffic, heavy traffic, packed road, high vehicle density, crowded road, many cars packed closely on road, wall-to-wall cars, overloaded road, lanes filled with vehicles

#### Queue

queue of cars, line of vehicles, cars backed up, long traffic queue, vehicle queue, cars waiting in line, traffic tailback, long line of stopped cars, intersection queue

#### Speed

slow-moving vehicles, crawling traffic, standstill traffic, stop-and-go traffic, vehicles barely moving, near-zero speed traffic, traffic at walking pace, reduced traffic speed

#### Moderate Congestion

slow but moving traffic, mild bottleneck, short vehicle queue, vehicles moving below normal speed, traffic slowing near intersection, cars moving with limited spacing, delayed traffic flow

#### Severe Congestion

Bumper-to-bumper traffic, gridlock, traffic jam, congested road, heavily congested, severe bottleneck, road completely filled with queued vehicles, rush-hour jam, all lanes blocked by queued vehicles

#### Visual Cues

brake lights visible, many red taillights, densely packed lanes, vehicles occupying all lanes, road full of cars, stationary vehicles in a long queue, little space between vehicles

#### Bottleneck Context

lane merge slowdown, toll booth queue, intersection overflow, road narrowing with queue, traffic buildup near exit, vehicles waiting at junction

## Class 4: Invalid Scene

Invalid Scene refers to frames where the traffic condition cannot be classified because the image is degraded, obstructed, irrelevant, non-traffic, or missing useful visual information.

### Invalid Scene — Seed Terms by Category

#### Camera Issue

severely blurred image, blurry photo where vehicles cannot be identified, out-of-focus image preventing traffic classification, motion blur obscuring vehicle movement, overexposed image, underexposed image, corrupted frame, distorted image, pixelated image

#### Visibility

scene too dark to determine traffic condition, nighttime frame where vehicles cannot be identified, poor visibility preventing traffic classification, fog so dense that vehicles cannot be identified, heavy rain completely obscuring road and vehicles, severe glare blocking most of the road view, sun glare covering traffic area, completely dark frame

#### Obstruction

blocked camera, obstructed view, covered lens, partial obstruction, camera malfunction, dirty lens, object in front of camera, camera fully blocked, lens covered by dirt, road hidden by obstruction

#### Semantic Failure

unclear traffic condition, non-traffic scene, irrelevant scene, unrecognizable image, empty frame, construction site with no vehicles, parking lot only with no active road traffic, indoor scene, no visible road area

#### Technical Error

black frame, white frame, static noise, camera off, no image data, frozen frame, camera disconnected, signal lost, corrupted CCTV frame, duplicated frozen frame

#### Framing Issue

road mostly outside the frame, only a small road corner visible, vehicles cut off by frame, camera pointed away from road, sky-only frame, dashboard interior only, building wall only, sidewalk-only scene
