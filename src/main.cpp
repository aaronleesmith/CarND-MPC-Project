#include <math.h>
#include <uWS/uWS.h>
#include <chrono>
#include <iostream>
#include <thread>
#include <vector>
#include "Eigen-3.3/Eigen/Core"
#include "Eigen-3.3/Eigen/QR"
#include "MPC.h"
#include "json.hpp"

// for convenience
using json = nlohmann::json;

/**
 * Helper functions
 */

constexpr double pi() { return M_PI; }

/**
 * Converts degrees to radians.
 * @param x
 * @return
 */
double deg2rad(double x) { return x * pi() / 180; }

/**
 * Converts radians to degrees.
 * @param x
 * @return
 */
double rad2deg(double x) { return x * 180 / pi(); }

/**
 * Checks if the JSON string from the simulator has data.
 * @param s
 * @return
 */
string hasData(string s) {
  auto found_null = s.find("null");
  auto b1 = s.find_first_of("[");
  auto b2 = s.rfind("}]");
  if (found_null != string::npos) {
    return "";
  } else if (b1 != string::npos && b2 != string::npos) {
    return s.substr(b1, b2 - b1 + 2);
  }
  return "";
}

/**
 * Evaluates a polynomial given coefficients and x.
 * @param coeffs
 * @param x
 * @return
 */
double polyeval(Eigen::VectorXd coeffs, double x) {
  double result = 0.0;
  for (int i = 0; i < coeffs.size(); i++) {
    result += coeffs[i] * pow(x, i);
  }
  return result;
}

/**
 * Fits a set of xvals and yvals to an order-order polynomial.
 * @param xvals
 * @param yvals
 * @param order
 * @return
 */
Eigen::VectorXd polyfit(Eigen::VectorXd xvals, Eigen::VectorXd yvals,
                        int order) {
  assert(xvals.size() == yvals.size());
  assert(order >= 1 && order <= xvals.size() - 1);
  Eigen::MatrixXd A(xvals.size(), order + 1);

  for (int i = 0; i < xvals.size(); i++) {
    A(i, 0) = 1.0;
  }

  for (int j = 0; j < xvals.size(); j++) {
    for (int i = 0; i < order; i++) {
      A(j, i + 1) = A(j, i) * xvals(j);
    }
  }

  auto Q = A.householderQr();
  auto result = Q.solve(yvals);
  return result;
}

/**
 * Converts a set of x, y points from the map coordinate system to the car coordinate system.
 * @param x
 * @param y
 * @param psi
 * @return [x, y]
 */
vector<double> map2Car(double x, double y, double psi) {
  return { x * cos(psi) + y * sin(psi), -x * sin(psi) + y * cos(psi) };
}

double mph2Mps(double mph_v) {
  return mph_v * (1609. / 3600.);
}

int main() {
  uWS::Hub h;
  double x = 10;
  // MPC is initialized here!
  MPC mpc;

  h.onMessage([&mpc](uWS::WebSocket<uWS::SERVER> ws, char *data, size_t length,
                     uWS::OpCode opCode) {
    // "42" at the start of the message means there's a websocket message event.
    // The 4 signifies a websocket message
    // The 2 signifies a websocket event
    string sdata = string(data).substr(0, length);
    if (sdata.size() > 2 && sdata[0] == '4' && sdata[1] == '2') {
      string s = hasData(sdata);
      if (s != "") {
        auto j = json::parse(s);
        string event = j[0].get<string>();
        if (event == "telemetry") {
          // j[1] is the data JSON object
          long long latency = 100; // In milliseconds.

          /**
           * The global x positions of the waypoints.
           */
          vector<double> waypoint_positions_x = j[1]["ptsx"];

          /**
           * The global y positions of the waypoints.
           */
          vector<double> waypoint_positions_y = j[1]["ptsy"];

          /**
           * The global x position of the vehicle.
           */
          double vehicle_global_x = j[1]["x"];

          /**
           * The global y position of the vehicle.
           */
          double vehicle_global_y = j[1]["y"];

          /**
           * The orientation of the vehicle in radians.
           */
          double vehicle_orientation = j[1]["psi"];

          /**
           * The current velocity in mph.
           */
          double velocity_in_mph = j[1]["speed"];

          /**
           * Placeholder variables for the MPC's estimated trajectory.
           */
          vector<double> mpc_x_vals;
          vector<double> mpc_y_vals;

          /**
           * Placeholder for the waypoints in the vehicles coordinate space (for display)
           */
          vector<double> next_x_vals(waypoint_positions_x.size());
          vector<double> next_y_vals(waypoint_positions_y.size());

          /**
           * We adjust the car's position and orientation forward based on the latency.
           */
          if (latency > 0) {
            double velocity_in_mps = mph2Mps(velocity_in_mph);
            vehicle_global_x += velocity_in_mps * cos(vehicle_orientation) * latency * .001; // Use seconds here instead of milliseconds.
            vehicle_global_y += velocity_in_mps * sin(vehicle_orientation) * latency * .001; // Use seconds here instead of milliseconds.
          }
          /**
           * Create a set of errors in the vehicles coordinate space.
           */
          Eigen::VectorXd vehicle_errors_in_vehicle_space_x(waypoint_positions_x.size());
          Eigen::VectorXd vehicle_errors_in_vehicle_space_y(waypoint_positions_y.size());

          for(int i = 0; i < waypoint_positions_x.size(); i++) {
            vector<double> vehicle_pos = map2Car(waypoint_positions_x[i] - vehicle_global_x,
                                                 waypoint_positions_y[i] - vehicle_global_y,
                                                 vehicle_orientation);

            vehicle_errors_in_vehicle_space_x[i] = vehicle_pos[0];
            vehicle_errors_in_vehicle_space_y[i] = vehicle_pos[1];

            next_x_vals[i] = vehicle_pos[0];
            next_y_vals[i] = vehicle_pos[1];
          }

          /**
           * Fit a line to the calculated errors.
           */
          int polynomial_order = 3;
          auto coeffs = polyfit(vehicle_errors_in_vehicle_space_x,
                                vehicle_errors_in_vehicle_space_y,
                                polynomial_order);

          /**
           * Create the state which will get sent to the solver.
           */
          Eigen::VectorXd state(6);
          state << 0,
                   0,
                   0,
                   velocity_in_mph,
                   polyeval(coeffs, 0.0),
                   -atan(coeffs[1]);

          double target_velocity_in_mph = 45;
          /**
           * Solve the non-linear problem using the MPC.
           */
          bool ok;
          auto vars = mpc.Solve(state, coeffs, mpc_x_vals, mpc_y_vals, target_velocity_in_mph, ok);

          double steer_value;
          double throttle_value;


          steer_value = vars[0];
          throttle_value = vars[1];


          json msgJson;
          msgJson["steering_angle"] = -steer_value;
          msgJson["throttle"] = throttle_value;

          //.. add (x,y) points to list here, points are in reference to the vehicle's coordinate system
          // the points in the simulator are connected by a Green line

          msgJson["mpc_x"] = mpc_x_vals;
          msgJson["mpc_y"] = mpc_y_vals;


          //.. add (x,y) points to list here, points are in reference to the vehicle's coordinate system
          // the points in the simulator are connected by a Yellow line

          msgJson["next_x"] = next_x_vals;
          msgJson["next_y"] = next_y_vals;


          auto msg = "42[\"steer\"," + msgJson.dump() + "]";
//          std::cout << msg << std::endl;
          // Latency
          // The purpose is to mimic real driving conditions where
          // the car does actuate the commands instantly.
          //
          // Feel free to play around with this value but should be to drive
          // around the track with 100ms latency.
          //
          // NOTE: REMEMBER TO SET THIS TO 100 MILLISECONDS BEFORE
          // SUBMITTING.
          // TO REVIEWER: LATENCY IS SET TO 100 AT THE TOP OF THIS METHOD.
          this_thread::sleep_for(chrono::milliseconds(latency));
          ws.send(msg.data(), msg.length(), uWS::OpCode::TEXT);
        }
      } else {
        // Manual driving
        std::string msg = "42[\"manual\",{}]";
        ws.send(msg.data(), msg.length(), uWS::OpCode::TEXT);
      }
    }
  });

  // We don't need this since we're not using HTTP but if it's removed the
  // program
  // doesn't compile :-(
  h.onHttpRequest([](uWS::HttpResponse *res, uWS::HttpRequest req, char *data,
                     size_t, size_t) {
    const std::string s = "<h1>Hello world!</h1>";
    if (req.getUrl().valueLength == 1) {
      res->end(s.data(), s.length());
    } else {
      // i guess this should be done more gracefully?
      res->end(nullptr, 0);
    }
  });

  h.onConnection([&h](uWS::WebSocket<uWS::SERVER> ws, uWS::HttpRequest req) {
    std::cout << "Connected!!!" << std::endl;
  });

  h.onDisconnection([&h](uWS::WebSocket<uWS::SERVER> ws, int code,
                         char *message, size_t length) {
    ws.close();
    std::cout << "Disconnected" << std::endl;
  });

  int port = 4567;
  if (h.listen(port)) {
    std::cout << "Listening to port " << port << std::endl;
  } else {
    std::cerr << "Failed to listen to port" << std::endl;
    return -1;
  }
  h.run();
}
