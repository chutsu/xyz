#!/usr/bin/octave -qf
graphics_toolkit("fltk");

function plot_measurements(plot_title, ts, data, yunit)
  hold on;
  plot(ts, data(1, :), "r-x", "linewidth", 2.0);
  plot(ts, data(2, :), "g-x", "linewidth", 2.0);
  plot(ts, data(3, :), "b-x", "linewidth", 2.0);
  title(plot_title)
	xlim([0, 0.5])
	xlabel("time [s]")
	ylabel(yunit)
	grid on;
  hold off;
endfunction

function compare_measurements(sensor, lerped_sensor, yunit)
  subplot(211);
  plot_measurements(cstrcat(sensor.title, " Data"),
                    sensor.ts,
                    sensor.data',
                    yunit);

  subplot(212);
  plot_measurements(cstrcat("Interpolated ", sensor.title, " Data"),
                    lerped_sensor.ts,
                    lerped_sensor.data,
                    yunit);
endfunction

function compare_timestamps(accel, gyro, lerped_accel, lerped_gyro)
  subplot(211);
  % -- Sensors
  accel_plot = 0.4;
  gyro_plot = 0.2;
  accel_y = accel_plot * ones(length(accel.ts), 1);
  gyro_y = gyro_plot * ones(length(gyro.ts), 1);
  % -- Plot sensor timestamps
  hold on;
  plot(accel.ts, accel_y, "bs", "linewidth", 2.0, "markerfacecolor", "b");
  plot(gyro.ts, gyro_y, "gs", "linewidth", 2.0, "markerfacecolor", "g");
  hold off;
  % -- Plot properties
  sensors = {"gyro", "accel"};
  y_tick = [gyro_plot, accel_plot];
  set(gca, "ytick", y_tick, "yticklabel", sensors);
  xlim([0.0, 0.01]);
  ylim([0.0, 0.6]);
  title("Raw data timestamps");
  xlabel("Time [s]");
  ylabel("Sensors");

  subplot(212);
  % -- Sensors
  accel_plot = 0.4;
  gyro_plot = 0.2;
  accel_y = accel_plot * ones(length(lerped_accel.ts), 1);
  gyro_y = gyro_plot * ones(length(lerped_gyro.ts), 1);
  % -- Plot sensor timestamps
  hold on;
  plot(lerped_accel.ts, accel_y, "bs", "linewidth", 2.0, "markerfacecolor", "b");
  plot(lerped_gyro.ts, gyro_y, "gs", "linewidth", 2.0, "markerfacecolor", "g");
  hold off;
  % -- Plot properties
  sensors = {"gyro", "accel"};
  y_tick = [gyro_plot, accel_plot];
  set(gca, "ytick", y_tick, "yticklabel", sensors);
  xlim([0.0, 0.01]);
  ylim([0.0, 0.6]);
  title("Interpolated data timestamps");
  xlabel("Time [s]");
  ylabel("Sensors");
endfunction

% Parse commandline args
arg_list = argv();
% -- Interpolated gyroscope
lerped_gyro = {};
lerped_gyro.ts = arg_list{1};
lerped_gyro.data = arg_list{2};
% -- Interpolated accelerometer
lerped_accel = {};
lerped_accel.ts = arg_list{3};
lerped_accel.data = arg_list{4};
% -- Raw gyroscope
gyro = {};
gyro.data = arg_list{5};
% -- Raw accelerometer
accel = {};
accel.data = arg_list{6};

% Load csv files
% -- Load interpolated gyroscope data
lerped_gyro.ts = csvread(lerped_gyro.ts);
lerped_gyro.data = transpose(csvread(lerped_gyro.data));
% -- Load interpolated accelerometer data
lerped_accel.ts = csvread(lerped_accel.ts);
lerped_accel.data = transpose(csvread(lerped_accel.data));
% -- Load raw gyroscope data
gyro.data = csvread(gyro.data);
gyro.ts = gyro.data(:, 1);
gyro.data = gyro.data(:, 2:4);
% -- Load raw accelerometer data
accel.data = csvread(accel.data);
accel.ts = accel.data(:, 1);
accel.data = accel.data(:, 2:4);

% Calculate relative time
t0 = min([min(lerped_gyro.ts),
          min(lerped_accel.ts),
          min(gyro.ts),
          min(accel.ts)]);
lerped_gyro.ts = (lerped_gyro.ts - t0) * 1e-9;
lerped_accel.ts = (lerped_accel.ts - t0) * 1e-9;
gyro.ts = (gyro.ts - t0) * 1e-9;
accel.ts = (accel.ts - t0) * 1e-9;

% Plot timestamps
figure(1);
compare_timestamps(accel, gyro, lerped_accel, lerped_gyro);

% Plot accelerometer measurements
figure(2);
sensor = {};
sensor.title = "Accelerometer";
sensor.ts = accel.ts;
sensor.data = accel.data;
yunit = "ms^-2";
compare_measurements(sensor, lerped_accel, yunit);

% % Plot gyroscope measurements
% figure(2);
% figure(3);
% sensor = {};
% sensor.title = "Gyroscope";
% sensor.ts = gyro.ts;
% sensor.data = gyro.data;
% yunit = "rad s^{-1}";
% compare_measurements(sensor, lerped_gyro, yunit);

ginput();
