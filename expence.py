import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from datetime import datetime

FILE_NAME = "expenses.csv.txt"

# Initialize CSV if not exists
try:
    df = pd.read_csv(FILE_NAME, parse_dates=['date'])
except FileNotFoundError:
    df = pd.DataFrame(columns=["date", "category", "amount"])
    df.to_csv(FILE_NAME, index=False)

# Load data
def load_data():
    return pd.read_csv(FILE_NAME, parse_dates=['date'])

# Save data
def save_data(df):
    df.to_csv(FILE_NAME, index=False)

# Add expense
def add_expense():
    date = input("Enter date (YYYY-MM-DD) or leave blank for today: ")
    if date == "":
        date = datetime.today().strftime('%Y-%m-%d')
    category = input("Enter category: ")
    amount = float(input("Enter amount: "))
    df = load_data()
    df = pd.concat([df, pd.DataFrame([{ "date": date, "category": category, "amount": amount }])], ignore_index=True)
    save_data(df)
    print("Expense added!")

# Delete expense
def delete_expense():
    df = load_data()
    if df.empty:
        print("No expenses to delete.")
        return
    
    # Show expenses grouped by day
    df['date_only'] = df['date'].dt.date
    grouped = df.groupby('date_only')
    print("\nExpenses by Day:")
    for date, group in grouped:
        print(f"\nDate: {date}")
        for idx, row in group.iterrows():
            print(f"  [{idx}] Category: {row['category']}, Amount: {row['amount']}")
    
    idx = int(input("\nEnter index of expense to delete: "))
    if idx in df.index:
        df = df.drop(idx).reset_index(drop=True)
        save_data(df)
        print("Expense deleted!")
    else:
        print("Invalid index!")

# View expenses grouped by day
def view_expenses():
    df = load_data()
    if df.empty:
        print("No expenses recorded.")
        return

    print("\nFull Data from expenses.csv.txt:\n")
    print(df)
    
    df['date_only'] = df['date'].dt.date
    grouped = df.groupby('date_only')

    print("\nExpenses by Day:")
    for date, group in grouped:
        print(f"\nDate: {date}")
        for idx, row in group.iterrows():
            print(f"  [{idx}] Category: {row['category']}, Amount: {row['amount']}")
        daily_total = group['amount'].sum()
        print(f"  --> Daily Total: {daily_total}")

    # Monthly summary
    monthly_total = df.groupby(df['date'].dt.to_period('M'))['amount'].sum()
    print("\nMonthly Total:\n", monthly_total)

# Show graph
def show_graph():
    df = load_data()
    if df.empty:
        print("No data to plot!")
        return
    
    choice = input("Graph by (daily/monthly/category): ").lower()
    
    if choice == "daily":
        data = df.groupby(df['date'].dt.date)['amount'].sum()
    elif choice == "monthly":
        data = df.groupby(df['date'].dt.to_period('M'))['amount'].sum()
    elif choice == "category":
        data = df.groupby('category')['amount'].sum()
    else:
        print("Invalid choice!")
        return
    
    data.plot(kind='bar', title=f'Expense {choice.capitalize()}')
    plt.ylabel('Amount')
    plt.show()

# Prediction using PyTorch and plot
class ExpensePredictor(nn.Module):
    def __init__(self):
        super(ExpensePredictor, self).__init__()
        self.fc = nn.Linear(1,1)  # Simple linear model

    def forward(self, x):
        return self.fc(x)

def predict_expense():
    df = load_data()
    if df.empty:
        print("No data for prediction!")
        return
    
    # Prepare data: group by day
    df_sorted = df.groupby(df['date'].dt.date)['amount'].sum().reset_index()
    df_sorted['day_num'] = range(len(df_sorted))
    x = torch.tensor(df_sorted['day_num'].values, dtype=torch.float32).view(-1,1)
    y = torch.tensor(df_sorted['amount'].values, dtype=torch.float32).view(-1,1)
    
    # Simple linear regression model
    model = ExpensePredictor()
    criterion = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    
    # Train model
    for epoch in range(500):
        optimizer.zero_grad()
        outputs = model(x)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()
    
    # Predict next 7 days
    future_days = 7
    x_future = torch.tensor([[i] for i in range(len(df_sorted), len(df_sorted)+future_days)], dtype=torch.float32)
    predicted = model(x_future).detach().numpy().flatten()
    
    # Combine historical and predicted for plotting
    all_days = list(df_sorted['date']) + [df_sorted['date'].iloc[-1] + pd.Timedelta(days=i) for i in range(1, future_days+1)]
    all_amounts = list(df_sorted['amount']) + list(predicted)
    
    # Plot
    plt.figure(figsize=(10,5))
    plt.plot(df_sorted['date'], df_sorted['amount'], marker='o', label='Actual')
    plt.plot(all_days[-future_days:], predicted, marker='x', linestyle='--', color='red', label='Predicted')
    plt.title("Expense Prediction for Next 7 Days")
    plt.xlabel("Date")
    plt.ylabel("Amount")
    plt.legend()
    plt.grid(True)
    plt.show()
    
    print("\nPredicted expenses for next 7 days:")
    for i in range(future_days):
        print(f"{all_days[-future_days + i]} : {predicted[i]:.2f}")

# Main menu
while True:
    print("\nPersonal Expense Tracker")
    print("1. Add Expense")
    print("2. Delete Expense")
    print("3. View Expenses")
    print("4. Show Graph")
    print("5. Predict Next 7 Days Expense")
    print("6. Exit")
    choice = input("Enter choice: ")
    
    if choice == "1":
        add_expense()
    elif choice == "2":
        delete_expense()
    elif choice == "3":
        view_expenses()
    elif choice == "4":
        show_graph()
    elif choice == "5":
        predict_expense()
    elif choice == "6":
        break
    else:
        print("Invalid choice!")

