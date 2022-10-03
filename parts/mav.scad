$fn = 20;

M2_screw_w = 2.7;
M2_nut_w = 4.9;
M2_nut_h = 2.0;

M3_screw_w = 3.2;
M3_caphead_w = 6.0;
M3_caphead_h = 3.0;
M3_nut_w = 6.5;
M3_nut_h = 2.5;

arm_l = 250;
arm_w = 12;
arm_inner_w = 10;

motor_mount_w = 26.0;
motor_mount_d = 26.0;
motor_mount_h = 6.0;

peg_inner_w = 30.0;
peg_outer_w = arm_l * 0.4;
peg_inner_screw_hole_w = 50.0;
peg_outer_screw_hole_w = 90.0;

frame_h = 4.0;
frame_standoff_w = 8.0;
frame_standoff_h = 10.0;
frame_support_w = 8.0;
frame_support_h = 5.0;



module frame(w, d, screw_w, nut_w, nut_h,
             standoff_w, standoff_h, support_w, support_h,
             nut_cst=0, nut_csb=0, disable=[]) {
  positions = [
    [w / 2, d / 2, standoff_h / 2],
    [w / 2, -d / 2, standoff_h / 2],
    [-w / 2, -d / 2, standoff_h / 2],
    [-w / 2, d / 2, standoff_h / 2]
  ];

  difference() {
    union() {
      // Standoffs
      supports = [0, 1, 2, 3];
      for (pos_idx = supports) {
        translate(positions[pos_idx]) {
          cylinder(r=standoff_w / 2.0, h=standoff_h, center=true);
        }
      }

      // Supports
      translate([0.0, d / 2, support_h / 2.0])
        cube([w, support_w, support_h], center=true);
      translate([0.0, -d / 2, support_h / 2.0])
        cube([w, support_w, support_h], center=true);
      translate([w / 2.0, 0.0, support_h / 2.0])
        cube([support_w, d, support_h], center=true);
      translate([-w / 2.0, 0.0, support_h / 2.0])
        cube([support_w, d, support_h], center=true);
    }

    // Mount Holes
    for (pos = positions) {
      translate(pos) {
        cylinder(r=screw_w / 2.0, h=standoff_h + 0.1, center=true);
      }
    }

    // Nut hole
    if (nut_cst || nut_csb) {
      for (pos = positions) {
        x = pos[0];
        y = pos[1];
        z = (nut_cst) ? standoff_h - nut_h / 2.0 : nut_h / 2.0;
        translate([x, y, z]) {
          cylinder(r=nut_w / 2.0, h=nut_h + 0.1, $fn=6, center=true);
        }
      }
    }

    // Disable standoffs
    for (pos_idx = disable) {
      x = positions[pos_idx][0];
      y = positions[pos_idx][1];
      z = positions[pos_idx][2] + (standoff_h - support_h) - 0.5;
      translate([x, y, z]) {
        cylinder(r=standoff_w / 2.0 + 0.01, h=support_h, center=true);
      }
    }
  }

  // Disable counter sinks
  for (pos_idx = disable) {
    x = positions[pos_idx][0];
    y = positions[pos_idx][1];
    z = (nut_cst) ? standoff_h - nut_h / 2.0 : nut_h / 2.0;
    translate([x, y, z]) {
      cylinder(r=nut_w / 2.0, h=nut_h, $fn=6, center=true);
    }
  }
}

module mav_motor() {
  color([1.0, 0.0, 0.0]) {
    // Shaft
    translate([0, 0, 25.7 + 14.0 / 2]) cylinder(r=6 / 2, h=14.0, center=true);

    // Body
    difference() {
      translate([0, 0, 25.7 / 2]) cylinder(r=27.9 / 2, h=25.7, center=true);

        translate([16 / 2, 0.0, 2.0 / 2.0])
            cylinder(r=M3_screw_w / 2, h=2.0 + 0.1, center=true);
        translate([-16 / 2, 0.0, 2.0 / 2.0])
            cylinder(r=M3_screw_w / 2, h=2.0 + 0.1, center=true);
        translate([0.0, 19.0 / 2.0, 2.0 / 2.0])
            cylinder(r=M3_screw_w / 2, h=2.0 + 0.1, center=true);
        translate([0.0, -19.0 / 2.0, 2.0 / 2.0])
            cylinder(r=M3_screw_w / 2, h=2.0 + 0.1, center=true);
    }

    // Bottom shaft
    h = 42.2 - (25.7 + 14.0);
    r = 2.0;
    translate([0.0, 0.0, -h / 2])
      cylinder(r=r, h=h + 0.1, center=true);
  }

  translate([0.0, 0.0, 25.7 + 14 - 12]) {
    color([0.0, 0.0, 0.0])
    scale(25.4)
      translate([0.0, 0.0, 0.0])
        rotate([90.0, 0.0, 0.0])
          import("../proto_parts/Propeller_1045/Propeller_1045.STL");
  }
}

module motor_mount(w, d, h, show_motor=1) {
  standoff_w = 8.0;
  standoff_h = 5.0;
  support_w = 3.5;
  support_h = 3.0;

  motor_hole_w = 19.0;
  motor_hole_d = 16.0;
  motor_hole_h = 6.0;

  // Show motor
  if (show_motor) {
    translate([0, 0, motor_hole_h + 0.01]) mav_motor();
  }

  // Motor mount
  difference() {
    // Mount body
    union() {
      // Motor mounts
      translate([0, motor_hole_w / 2, motor_hole_h / 2])
        cylinder(r=4, h=motor_hole_h, center=true);
      translate([0, -motor_hole_w / 2, motor_hole_h / 2])
        cylinder(r=4, h=motor_hole_h, center=true);
      translate([motor_hole_d / 2, 0, motor_hole_h / 2])
        cylinder(r=4, h=motor_hole_h, center=true);
      translate([-motor_hole_d / 2, 0, motor_hole_h / 2])
        cylinder(r=4, h=motor_hole_h, center=true);

      // Supports
      translate([0, 0, support_h / 2])
        cube([motor_hole_d, 8, support_h], center=true);
      translate([0, 0, support_h / 2])
        cube([8, motor_hole_w, support_h], center=true);

      // Arm
      cube([motor_hole_d + 10, arm_w, motor_hole_h], center=true);
      translate([0, 0, -arm_w / 4])
        cube([motor_hole_d + 10, arm_w, arm_w / 2], center=true);
    }

    // Mount holes
    translate([0, motor_hole_w / 2, motor_hole_h / 2])
      cylinder(r=M3_screw_w / 2, h=motor_hole_h + 0.01, center=true);
    translate([0, -motor_hole_w / 2, motor_hole_h / 2])
      cylinder(r=M3_screw_w / 2, h=motor_hole_h + 0.01, center=true);
    translate([motor_hole_d / 2, 0, motor_hole_h / 2])
      cylinder(r=M3_screw_w / 2, h=motor_hole_h + 1, center=true);
    translate([-motor_hole_d / 2, 0, motor_hole_h / 2])
      cylinder(r=M3_screw_w / 2, h=motor_hole_h + 1, center=true);

    // Mount hole conter sink
    translate([0, motor_hole_w / 2, M3_caphead_h / 2])
      cylinder(r=M3_caphead_w / 2, h=M3_caphead_h + 0.1, center=true);
    translate([0, -motor_hole_w / 2, M3_caphead_h / 2])
      cylinder(r=M3_caphead_w / 2, h=M3_caphead_h + 0.1, center=true);

    // Arm divit
    translate([0, 0, -arm_w / 2])
      rotate([90, 0, 90])
        cylinder(r=arm_w / 2, h=motor_hole_d + 10 + 0.01, center=true);
    translate([0, 0, -5])
      cube([motor_hole_w + 10, motor_hole_d + 10, 2.0], center=true);
  }
}

module motor_mount_spacer(w, d, h, show_motor=1) {
  standoff_w = 8.0;
  standoff_h = 5.0;
  support_w = 3.5;
  support_h = 3.0;

  motor_hole_w = 19.0;
  motor_hole_d = 16.0;
  motor_hole_h = 6.0;

  difference() {
    // Body
    union() {
      cube([arm_w, motor_hole_d + 10, motor_hole_h], center=true);
    }

    // Arm divit
    translate([0, 0, arm_w / 2])
      rotate([90, 0, 0])
        cylinder(r=arm_w / 2, h=motor_hole_d + 10 + 0.01, center=true);

    translate([0, motor_hole_w / 2, -motor_hole_h / 4])
      cylinder(r=M3_screw_w / 2, h=motor_hole_h / 2 + 1, center=true);
    translate([0, -motor_hole_w / 2, -motor_hole_h / 4])
      cylinder(r=M3_screw_w / 2, h=motor_hole_h / 2 + 1, center=true);
  }
}

module mav_arm(arm_w, arm_l, show_motor=1) {
  // Show motor
  if (show_motor) {
    y_offset = arm_l / 2 - motor_mount_d / 2;

    translate([0, y_offset, arm_w])
      rotate(90)
        motor_mount(motor_mount_w, motor_mount_d, motor_mount_h, show_motor=1);

    translate([0, y_offset, 0])
      rotate(0)
        motor_mount_spacer(motor_mount_w, motor_mount_d, motor_mount_h);
  }


  // Arm
  color([0.2, 0.2, 0.2])
  translate([0.0, 0.0, arm_w / 2.0]) {
    rotate([90.0, 0.0, 0.0]) {
      difference() {
        cylinder(r=arm_w / 2, h=arm_l, center=true);
        cylinder(r=arm_inner_w / 2, h=arm_l + 0.01, center=true);

        // cube([arm_w, arm_w, arm_l], center=true);
        // cube([arm_inner_w, arm_inner_w, arm_l + 0.01], center=true);
      }
    }
  }
}

module mav_arm_peg(peg_inner_w, peg_outer_w, peg_inner_screw_hole_w, peg_outer_screw_hole_w) {
  tol = 0.5;

  color([0, 0, 1]) {
    difference() {
      union() {
        // // Center
        // translate([0, 0, arm_w / 2])
        //   cube([arm_w, peg_inner_w, arm_w], center=true);
        // translate([0, 0, arm_w / 2])
        //   cube([peg_inner_w, arm_w, arm_w], center=true);

        // Pegs
        translate([0, 0, arm_w / 2])
          rotate([90, 0, 0])
            cylinder(r=(arm_inner_w - tol) / 2, h=peg_outer_w, center=true);
        translate([0, 0, arm_w / 2])
          rotate([90, 0, 90])
            cylinder(r=(arm_inner_w - tol) / 2, h=peg_outer_w, center=true);
      }

      // Screw hole
      for (i = [1:4]) {
        rotate(90.0 * i) {
          translate([peg_inner_screw_hole_w / 2, 0.0, arm_w / 2])
            cylinder(r=M3_screw_w / 2, h=arm_w + 0.01, center=true);

          translate([peg_outer_screw_hole_w / 2, 0.0, arm_w / 2])
            cylinder(r=M3_screw_w / 2, h=arm_w + 0.01, center=true);
        }
      }
    }
  }
}

module mav_arm_supports(counter_sink_type=0) {
  support_l = arm_l * 0.45;
  support_h = 4.0;
  holes_outer = 50.0;

  difference() {
    union() {
      translate([0, 0, support_h / 2])
        cube([arm_w, support_l, support_h], center=true);
      translate([0, 0, support_h / 2])
        cube([support_l, arm_w, support_h], center=true);
      translate([0, 0, -arm_w / 4])
        cube([arm_w, support_l, arm_w / 2], center=true);
      translate([0, 0, -arm_w / 4])
        cube([support_l, arm_w, arm_w / 2], center=true);
    }

    for (i = [1:4]) {
      rotate(90 * i) {
        translate([peg_inner_screw_hole_w / 2, 0.0, support_h / 2])
          cylinder(r=M3_screw_w / 2, h=support_h + 0.01, center=true);
        translate([peg_outer_screw_hole_w / 2, 0.0, support_h / 2])
          cylinder(r=M3_screw_w / 2, h=support_h + 0.01, center=true);

        if (counter_sink_type == 1) {
          // Hex counter sink
          translate([peg_inner_screw_hole_w / 2, 0.0, support_h - M3_nut_h / 2])
            cylinder(r=M3_nut_w / 2, h=M3_nut_h + 0.01, $fn=6, center=true);
          // translate([peg_outer_screw_hole_w / 2, 0.0, support_h - M3_nut_h / 2])
          //   cylinder(r=M3_nut_w / 2, h=M3_nut_h + 0.01, $fn=6, center=true);

        } else if (counter_sink_type == 0) {
          // Caphead counter sink
          translate([peg_inner_screw_hole_w / 2, 0.0, support_h - M3_caphead_h / 2])
            cylinder(r=M3_caphead_w / 2, h=M3_caphead_h + 0.01, center=true);
          // translate([peg_outer_screw_hole_w / 2, 0.0, support_h - M3_caphead_h / 2])
          //   cylinder(r=M3_caphead_w / 2, h=M3_caphead_h + 0.01, center=true);
        }
      }
    }

    rotate([90, 0, 0])
      translate([0, -arm_w / 2, 0])
        cylinder(r=(arm_w + 0.5) / 2, h=support_l + 0.1, center=true);
    rotate([90, 0, 90])
      translate([0, -arm_w / 2, 0])
        cylinder(r=(arm_w + 0.5) / 2, h=support_l + 0.1, center=true);
    translate([0, 0, -support_h / 2 - 3])
      cube([support_l + 1, support_l + 1, support_h], center=true);
  }
}

module mav_frame(frame_standoff_w, frame_standoff_h,
                 frame_support_w, frame_support_h) {
  outer_w = peg_outer_screw_hole_w;
  mount_w = sqrt(pow(outer_w, 2) + pow(outer_w, 2)) / 2;
  frame_w = 90.0;
  frame_d = 40.0;

  difference() {
    union() {
      // Mount frame
      frame(mount_w, mount_w,
            M3_screw_w, M3_nut_w, M3_nut_h,
            frame_standoff_w, frame_support_h,
            frame_support_w, frame_support_h);

      // Mav frame
      frame(frame_d, frame_w,
            M3_screw_w, M3_nut_w, M3_nut_h,
            frame_standoff_w, frame_standoff_h,
            frame_support_w, frame_support_h);
    }

    // Screw counter sink
    translate([mount_w / 2, mount_w / 2, frame_support_h - M3_caphead_h / 2])
      cylinder(r=M3_caphead_w / 2, h=M3_caphead_h + 0.01, center=true);
    translate([-mount_w / 2, mount_w / 2, frame_support_h - M3_caphead_h / 2])
      cylinder(r=M3_caphead_w / 2, h=M3_caphead_h + 0.01, center=true);
    translate([mount_w / 2, -mount_w / 2, frame_support_h - M3_caphead_h / 2])
      cylinder(r=M3_caphead_w / 2, h=M3_caphead_h + 0.01, center=true);
    translate([-mount_w / 2, -mount_w / 2, frame_support_h - M3_caphead_h / 2])
      cylinder(r=M3_caphead_w / 2, h=M3_caphead_h + 0.01, center=true);

    // Screw counter sink
    translate([frame_d / 2, frame_w / 2, M3_nut_h / 2])
      cylinder(r=M3_nut_w / 2, h=M3_nut_h + 0.01, $fn=6, center=true);
    translate([-frame_d / 2, frame_w / 2, M3_nut_h / 2])
      cylinder(r=M3_nut_w / 2, h=M3_nut_h + 0.01, $fn=6, center=true);
    translate([frame_d / 2, -frame_w / 2, M3_nut_h / 2])
      cylinder(r=M3_nut_w / 2, h=M3_nut_h + 0.01, $fn=6, center=true);
    translate([-frame_d / 2, -frame_w / 2, M3_nut_h / 2])
      cylinder(r=M3_nut_w / 2, h=M3_nut_h + 0.01, $fn=6, center=true);
  }
}

module mav_payload_frame(frame_standoff_w, frame_standoff_h,
                         frame_support_w, frame_support_h,
                         nut_counter_sink=1) {
  outer_w = peg_outer_screw_hole_w;
  mount_w = sqrt(pow(outer_w, 2) + pow(outer_w, 2)) / 2;
  frame_w = 90.0;
  frame_d = 40.0;

  difference() {
    union() {
      // Mount frame
      frame(mount_w, mount_w,
            M3_screw_w, M3_nut_w, M3_nut_h,
            frame_standoff_w, frame_support_h,
            frame_support_w, frame_support_h);

      // Mav frame
      frame(frame_d, frame_w,
            M3_screw_w, M3_nut_w, M3_nut_h,
            frame_standoff_w, frame_standoff_h,
            frame_support_w, frame_support_h);
    }

    if (nut_counter_sink) {
      // Nut counter sink
      translate([mount_w / 2, mount_w / 2, frame_support_h - M3_nut_h / 2])
        cylinder(r=M3_nut_w / 2, h=M3_nut_h + 0.01, $fn=6, center=true);
      translate([-mount_w / 2, mount_w / 2, frame_support_h - M3_nut_h / 2])
        cylinder(r=M3_nut_w / 2, h=M3_nut_h + 0.01, $fn=6, center=true);
      translate([mount_w / 2, -mount_w / 2, frame_support_h - M3_nut_h / 2])
        cylinder(r=M3_nut_w / 2, h=M3_nut_h + 0.01, $fn=6, center=true);
      translate([-mount_w / 2, -mount_w / 2, frame_support_h - M3_nut_h / 2])
        cylinder(r=M3_nut_w / 2, h=M3_nut_h + 0.01, $fn=6, center=true);
    }
  }
}

module mav_assembly() {
  // Arm center peg
  rotate(45)
    mav_arm_peg(peg_inner_w, peg_outer_w, peg_inner_screw_hole_w, peg_outer_screw_hole_w);

  // Arms
  for (i = [0:4]) {
    rotate(45.0 + 90.0 * i)
      translate([0.0, arm_l / 2.0 + 15.0, 0.0])
        mav_arm(arm_w, arm_l);
  }

  // Arm supports
  rotate(45) {
    // Top support
    translate([0, 0, arm_w + 0.01])
      mav_arm_supports(0);

    // Bottom support
    rotate([180, 0, 0])
      translate([0, 0, 0.01])
        mav_arm_supports(1);
  }

  // Frame
  // -- Top Frame
  translate([0.0, 0.0, 4 + arm_w + 0.1])
    rotate(90)
      mav_frame(frame_standoff_w, frame_standoff_h,
                frame_support_w, frame_support_h);
  // -- Bottom Frame
  translate([0.0, 0.0, -4 + 0.1])
    rotate([180, 0, 0])
      mav_payload_frame(frame_standoff_w, frame_standoff_h,
                        frame_support_w, frame_support_h);
}

module print() {
  // Motor mounts and mount
  for (i = [0:3]) {
    x = 40 * i;
    // Motor mounts
    translate([x, 0, 0])
      mav_motor_mount(motor_mount_w, motor_mount_d, motor_mount_h, arm_w,
                      mode=0,
                      show_motor=0);

    // Motor mount
    translate([x, 80, 0])
      motor_mount(motor_mount_w, motor_mount_d, motor_mount_h, show_motor=0);
  }
}

// Main
// print();
mav_assembly();

// Develop
// motor_mount(motor_mount_w, motor_mount_d, motor_mount_h, show_motor=1);
// motor_mount_spacer(motor_mount_w, motor_mount_d, motor_mount_h, show_motor=1);
// mav_arm(arm_w, arm_l);
// mav_arm_peg(peg_inner_w, peg_outer_w,
//             peg_inner_screw_hole_w, peg_outer_screw_hole_w);
// mav_arm_supports();
// mav_frame(frame_standoff_w, frame_standoff_h,
//           frame_support_w, frame_support_h);
