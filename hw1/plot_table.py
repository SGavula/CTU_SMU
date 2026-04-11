import numpy as np
import matplotlib.pyplot as plt

# Action mapping
def best_action(values):
    return np.argmax(values)  # 0 = stand, 1 = hit

def action_symbol(a):
    return "S" if a == 0 else "H"

# Plotting function
def plot_table(table, row_labels, dealer_cards, title):
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.axis('off')

    tbl = ax.table(
        cellText=table,
        rowLabels=row_labels,
        colLabels=dealer_cards,
        loc='center'
    )

    # Color cells
    for i in range(table.shape[0]):
        for j in range(table.shape[1]):
            val = table[i, j]
            cell = tbl[i+1, j]  # +1 because of header

            if val == "S":
                cell.set_facecolor("red")
            elif val == "H":
                cell.set_facecolor("green")

    tbl.scale(1, 1.5)
    plt.title(title, pad=20, fontsize=16)

    # Add dealer's card label above the columns
    dealer_pos = 1 if title == "Hard Totals" else 0.82
    plt.text(0.5, dealer_pos, "Dealer's Upcard", transform=ax.transAxes, 
             ha='center', fontsize=10)

    # Add player's hand label to the left of the rows
    plt.text(-0.05, 0.5, "Player's Hand", transform=ax.transAxes, 
             va='center', rotation='vertical', fontsize=10)

    plt.show()


def plot_tables(Q):
    # Dealer cards (columns)
    dealer_cards = list(range(2, 12))  # 2–10 + Ace(11)

    # Player sums (rows)
    hard_sums = list(range(21, 4, -1))   # 5–21
    soft_sums = list(range(21, 11, -1))  # A,A = 12, A,2 = 13 ... A,10 = 21


    # Create tables
    hard_table = np.empty((len(hard_sums), len(dealer_cards)), dtype=object)
    soft_table = np.empty((len(soft_sums), len(dealer_cards)), dtype=object)


    # Fill HARD table
    for i, ps in enumerate(hard_sums):
        for j, dc in enumerate(dealer_cards):
            state = (ps, dc, False)
            if state in Q:
                a = best_action(Q[state])
                hard_table[i, j] = action_symbol(a)
            else:
                hard_table[i, j] = ""


    # Fill SOFT table
    for i, ps in enumerate(soft_sums):
        for j, dc in enumerate(dealer_cards):
            state = (ps, dc, True)
            if state in Q:
                a = best_action(Q[state])
                soft_table[i, j] = action_symbol(a)
            else:
                soft_table[i, j] = ""

    # Plot both
    plot_table(hard_table, hard_sums, dealer_cards, "Hard Totals")
    plot_table(soft_table, soft_sums, dealer_cards, "Soft Totals")