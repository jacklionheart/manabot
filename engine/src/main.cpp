// src/main.cpp
#include <iostream>
#include <memory>
#include <map>
#include "graphics/game_display.h"
#include "rules/engine/game.h"
#include "rules/engine/player.h"
#include "rules/cards/card.h"  // Include this to use Card
#include "cardsets/card_registry.h"
#include <thread>

int main() {
    // Initialize the game
    registerAllCards();

    // Define decklists using card quantities
    Decklist red_player_cards = {
        {"Mountain", 3},
        {"Grey Ogre", 4}
    };

    Decklist green_player_cards = {
        {"Forest", 2},
        {"Llanowar Elves", 5}
    };

    std::vector<Decklist> decklists = {red_player_cards, green_player_cards};
    
    // Create players
    std::vector<PlayerConfig> players;
    players.emplace_back(new PlayerConfig("Red Player", red_player_cards));
    players.emplace_back(new PlayerConfig("Green Player", green_player_cards));

    // Initialize the game
    Game game(players);
    game.play();

    std::cout << "Game has ended after " << game->global_turn_count << " turns." << std::endl;

    return 0;
}