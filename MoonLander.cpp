#include <SFML/Graphics.hpp>
#include <armadillo>
#include <cmath>
#include <random>
#include <iostream>
#include <tuple>

class MoonLanderEnv {
public:
    int input_size = 8;
    int  output_size = 4;
    MoonLanderEnv(int width = 600, int height = 400, bool render_mode = false)
        : width(width), height(height), render_mode(render_mode),
          lander_width(20), lander_height(30),
          gravity(0.08), thrust(0.25), horizontal_thrust(0.15),
          landing_pad_width(80), landing_pad_height(10),
          input_features(8), output_features(4), frame_iteration(0)
    {

        if (render_mode) {
            window = new sf::RenderWindow(sf::VideoMode(width, height), "Moon Lander Environment");
        }


        rng.seed(std::random_device()());
        landing_pad_dist = std::uniform_int_distribution<int>(landing_pad_width / 2, width - landing_pad_width / 2);
        reset();
    }

    ~MoonLanderEnv() {
        if (render_mode && window != nullptr) {
            window->close();
            delete window;
        }
    }


    arma::vec reset() {
        lander_x = width / 2.0;
        lander_y = 50.0;
        lander_vel_x = 0.0;
        lander_vel_y = 0.0;
        angle = 0.0;
        angular_velocity = 0.0;
        landing_pad_x = landing_pad_dist(rng);
        landing_pad_y = height - landing_pad_height;
        crashed = false;
        landed = false;
        leg_l_contact = false;
        leg_r_contact = false;
        frame_iteration = 0;
        return getState();
    }


    std::tuple<arma::vec, double, bool> step(int action) {
        double reward = -1.0;
        bool done = false;
        frame_iteration++;


        if (action == 1) { 
            lander_vel_y -= thrust * std::cos(angle * M_PI / 180.0);
            lander_vel_x += thrust * std::sin(angle * M_PI / 180.0);
        } else if (action == 2) { 
            angular_velocity += 2;
        } else if (action == 3) { 
            angular_velocity -= 2;
        }


        lander_vel_y += gravity;


        lander_x += lander_vel_x;
        lander_y += lander_vel_y;


        angle += angular_velocity;

        if (angle >= 360) {
            angle = std::fmod(angle, 360);
        } else if (angle < 0) {
            angle = 360 - std::fmod(-angle, 360);
        }
        angular_velocity *= 0.9;


        sf::FloatRect landerRect(lander_x - lander_width / 2.0f, lander_y - lander_height / 2.0f, lander_width, lander_height);
        sf::FloatRect landingPadRect(landing_pad_x - landing_pad_width / 2.0f, landing_pad_y, landing_pad_width, landing_pad_height);


        leg_l_contact = false;
        leg_r_contact = false;

        double angle_rad = angle * M_PI / 180.0;

        double leg_l_y = lander_y + (lander_height / 2.0) * std::cos(angle_rad) + (lander_width / 2.0) * std::sin(angle_rad);
        double leg_r_y = lander_y + (lander_height / 2.0) * std::cos(angle_rad) - (lander_width / 2.0) * std::sin(angle_rad);

        if (leg_l_y >= landing_pad_y &&
            landerRect.left >= landingPadRect.left &&
            landerRect.left <= landingPadRect.left + landingPadRect.width) {
            leg_l_contact = true;
        }
        if (leg_r_y >= landing_pad_y &&
            (landerRect.left + landerRect.width) >= landingPadRect.left &&
            (landerRect.left + landerRect.width) <= landingPadRect.left + landingPadRect.width) {
            leg_r_contact = true;
        }

        if (lander_y + lander_height >= height) {
            if (landerRect.intersects(landingPadRect) && leg_l_contact && leg_r_contact) {
                if (std::abs(lander_vel_x) < 1.0 && std::abs(lander_vel_y) < 2.0 && std::abs(angle) < 10.0) {
                    reward = 1000.0;
                    landed = true;
                    done = true;
                } else {
                    reward = -100.0;
                    crashed = true;
                    done = true;
                }
            } else {
                reward = -100.0;
                crashed = true;
                done = true;
            }
        }

        if (lander_x < 0 || lander_x > width) {
            reward = -100.0;
            crashed = true;
            done = true;
        }

        if (render_mode) {
            render();
        }

        return std::make_tuple(getState(), reward, done);
    }


    void render() {
        if (!render_mode || !window) return;

        window->clear(sf::Color::Black);


        sf::RectangleShape landingPad(sf::Vector2f(landing_pad_width, landing_pad_height));
        landingPad.setFillColor(sf::Color::Green);
        landingPad.setPosition(landing_pad_x - landing_pad_width / 2.0f, landing_pad_y);
        window->draw(landingPad);


        sf::ConvexShape lander;
        lander.setPointCount(3);
        lander.setPoint(0, sf::Vector2f(0, lander_height));
        lander.setPoint(1, sf::Vector2f(lander_width / 2.0f, 0));
        lander.setPoint(2, sf::Vector2f(lander_width, lander_height));
        lander.setFillColor(sf::Color::White);

        lander.setOrigin(lander_width / 2.0f, lander_height / 2.0f);
        lander.setPosition(lander_x, lander_y);

        lander.setRotation(-angle);
        window->draw(lander);

        window->display();
        sf::sleep(sf::milliseconds(20));

        sf::Event event;
        while (window->pollEvent(event)) {
            if (event.type == sf::Event::Closed) {
                window->close();
                exit(0);
            }
        }
    }

private:
    int width, height;
    bool render_mode;
    double lander_width, lander_height;
    double gravity, thrust, horizontal_thrust;
    double landing_pad_width, landing_pad_height;
    int input_features, output_features;


    double lander_x, lander_y;
    double lander_vel_x, lander_vel_y;
    double angle, angular_velocity;
    double landing_pad_x, landing_pad_y;
    bool crashed, landed;
    bool leg_l_contact, leg_r_contact;
    int frame_iteration;

    sf::RenderWindow* window = nullptr;


    std::mt19937 rng;
    std::uniform_int_distribution<int> landing_pad_dist;

    arma::vec getState() {
        arma::vec state(8);
        state(0) = lander_x / width;
        state(1) = lander_y / height;
        state(2) = lander_vel_x / 5.0;
        state(3) = lander_vel_y / 5.0;
        state(4) = angle / 180.0;
        state(5) = angular_velocity / 5.0;
        state(6) = leg_l_contact ? 1.0 : 0.0;
        state(7) = leg_r_contact ? 1.0 : 0.0;
        return state;
    }
};


