// stack.cpp
#include "stack.h"
#include "rules/game.h"

#include <format>

#include <spdlog/spdlog.h>

Stack::Stack(Game* game)
    : Zone(game) {}

void Stack::move(Card* card) {
    // simple alias for now. Maybe remove one day?
    this->cast(card);
}

void Stack::remove(Card* card) {
    Zone::remove(card);
    objects.erase(std::remove_if(objects.begin(), objects.end(),
        [&card](const std::unique_ptr<Spell>& spell) { return spell->card == card; }),
        objects.end()); 
}

void Stack::cast(Card* card) {
    if (card->current_zone != NULL && card->current_zone == this) {
        throw std::logic_error(std::format("Card {} already on the stack", card->toString()));
    }
    Zone::move(card);
    objects.emplace_back(new Spell(card));
}

void Stack::resolveTop() {
    if (!objects.empty()) {
        std::unique_ptr<StackObject> object = std::move(objects.back());
        objects.pop_back();
        object->resolve(game);
    }
}

size_t Stack::size() const {
    return objects.size();
}

int Spell::next_id = 0;

Spell::Spell(Card* card)
    : id(next_id++), card(card), controller(card->owner) {}

void Spell::resolve(Game* game) {
    if (card->types.isPermanent()) {
        game->zones->battlefield->enter(card);
    } else {
        // Implement logic for instants and sorceries
        // For now, we simply log the event
    }
}