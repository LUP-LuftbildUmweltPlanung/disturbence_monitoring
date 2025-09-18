def hold_point(config, message="Ergebnisse prüfen. Weiter mit Enter, Abbruch mit 'n': "):
    if config.get("hold", False):
        user_input = input(message)
        if user_input.lower() == "n":
            print("Abgebrochen durch Benutzer.")
            exit(0)
