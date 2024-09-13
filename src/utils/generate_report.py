def save_report_as_html(accuracy_rf, accuracy_lr):
    with open("reporte_final.html", "w") as file:
        file.write(f"<h1>Reporte Final</h1>")
        file.write(f"<h2>Resultados de Modelos</h2>")
        file.write(f"<p>Exactitud de Random Forest: {accuracy_rf}</p>")
        file.write(f"<p>Exactitud de Logistic Regression: {accuracy_lr}</p>")
        file.write(f"<h2>Conclusiones</h2>")
        mejor_modelo = 'Random Forest' if accuracy_rf > accuracy_lr else 'Logistic Regression'
        file.write(f"<p>El modelo más preciso fue: {mejor_modelo}</p>")

