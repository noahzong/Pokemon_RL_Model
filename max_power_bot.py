import asyncio
import time
from tabulate import tabulate
from poke_env.player import Player, RandomPlayer, cross_evaluate


class MaxBasePowerPlayer(Player):
    def choose_move(self, battle):
        # If the player can attack, it will
        if battle.available_moves:
            # Finds the best move among available ones
            best_move = max(battle.available_moves, key=lambda move: move.base_power)
            return self.create_order(best_move)

        # If no attack is available, a random switch will be made
        else:
            return self.choose_random_move(battle)


async def main():
    start = time.time()

    # We create two players.
    random_player = RandomPlayer(battle_format="gen8randombattle")
    max_base_power_player = MaxBasePowerPlayer(battle_format="gen8randombattle")

    # Now, let's evaluate our player
    await max_base_power_player.battle_against(random_player, n_battles=10)

    print(
        "Max Base Power Player won %d / 10 battles [this took %f seconds]"
        % (max_base_power_player.n_won_battles, time.time() - start)
    )

    #Create array of players
    players = [random_player, max_base_power_player]

    # Now, we can cross evaluate them: every player will play 10 games against every
    # other player.
    cross_evaluation = await cross_evaluate(players, n_challenges=10)

    # Defines a header for displaying results
    table = [["-"] + [p.username for p in players]]

    # Adds one line per player with corresponding results
    for p_1, results in cross_evaluation.items():
        table.append([p_1] + [cross_evaluation[p_1][p_2] for p_2 in results])

    # Displays results in a nicely formatted table.
    print(tabulate(table))


if __name__ == "__main__":
    asyncio.get_event_loop().run_until_complete(main())