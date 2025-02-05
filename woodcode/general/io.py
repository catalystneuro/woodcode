from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import Select
from getpass import getpass, GetPassWarning
import time


def get_tickatlab_info(animal_id, username=None, password=None):
    """
    Get information about an animal from tick@lab database.

    Parameters:
    -----------
    animal_id : str
        The ID number of the animal to look up
    username : str, optional
        Your tick@lab username. If not provided, will prompt for input
    password : str, optional
        Your tick@lab password. If not provided, will prompt for input

    Returns:
    --------
    dict
        Dictionary containing the animal's information, or None if lookup failed
    """
    print("Looking up tick@lab")
    url = "https://www.resource-manager.brr.mvm.ed.ac.uk/tickatlab/default.aspx"
    driver = None

    # Get credentials if not provided
    if username is None:
        username = input("Enter your tick@lab username: ")
    if password is None:
        try:
            password = getpass("Enter your tick@lab password: ")
        except GetPassWarning:
            print("Note: Password will be visible when typing")
            password = input("Enter your tick@lab password: ")

    try:
        # Setup Chrome options
        chrome_options = Options()
        chrome_options.add_argument('--headless')
        driver = webdriver.Chrome(options=chrome_options)
        wait = WebDriverWait(driver, 20)

        # Login process
        driver.get(url)
        username_field = wait.until(EC.presence_of_element_located((By.ID, "login")))
        username_field.send_keys(username)
        submit_button = wait.until(EC.element_to_be_clickable((By.ID, "submit")))
        submit_button.click()

        password_field = wait.until(EC.presence_of_element_located((By.ID, "password")))
        password_field.send_keys(password)
        submit_button = wait.until(EC.element_to_be_clickable((By.ID, "submit")))
        submit_button.click()

        # Navigate to breeding page
        breeding_link = wait.until(EC.element_to_be_clickable((By.ID, "db180813MWE7")))
        driver.execute_script("arguments[0].click();", breeding_link)

        # Set display filter and search
        display_dropdown = wait.until(EC.presence_of_element_located((By.ID, "A100503DBE01--1S1")))
        select = Select(display_dropdown)
        select.select_by_value("STOCKANDEXITED")
        time.sleep(1)

        # Enter animal ID
        id_field = wait.until(EC.presence_of_element_located((By.ID, "A100928MWE1--1S1")))
        id_field.clear()
        id_field.send_keys(str(animal_id))

        # Click Apply Filter
        apply_button = wait.until(EC.presence_of_element_located((
            By.XPATH, "//span[@class='USEFILTER button__label']/parent::*"
        )))
        driver.execute_script("arguments[0].click();", apply_button)

        # Wait for results
        time.sleep(2)

        # Find the data row that contains our animal
        rows = driver.find_elements(By.CSS_SELECTOR, "tr.rowstyle, tr.altrowstyle")
        target_row = None

        for row in rows:
            cells = row.find_elements(By.TAG_NAME, "td")
            if len(cells) > 0 and str(animal_id) in cells[0].text:
                target_row = cells
                break

        if not target_row:
            print(f"No data found for animal ID: {animal_id}")
            return None

        # Extract data with corrected indices
        # Print all cell contents for debugging
        print("\nDebug - All cell contents:")
        for i, cell in enumerate(target_row):
            print(f"Cell {i}: {cell.text}")

        animal_data = {
            'Animal_ID': target_row[0].text if len(target_row) > 0 else "N/A",
            'Cage_ID': target_row[2].text if len(target_row) > 2 else "N/A",
            'Strain': target_row[3].text if len(target_row) > 3 else "N/A",
            'DoB': target_row[5].text if len(target_row) > 5 else "N/A",
            'Age': target_row[6].text if len(target_row) > 6 else "N/A",
            'No_of_animals': target_row[8].text if len(target_row) > 8 else "N/A",
            'Sire': target_row[9].text if len(target_row) > 9 else "N/A",
            'Dam': target_row[10].text if len(target_row) > 10 else "N/A",
            'Team': target_row[11].text if len(target_row) > 11 else "N/A",
            'PPL': target_row[12].text if len(target_row) > 12 else "N/A",
            'Protocol': target_row[13].text if len(target_row) > 13 else "N/A",
            'Room': target_row[14].text if len(target_row) > 14 else "N/A",
            'Status': target_row[15].text if len(target_row) > 15 else "N/A",
            'Project_code': target_row[16].text if len(target_row) > 16 else "N/A",
            'Responsible_User': target_row[18].text if len(target_row) > 18 else "N/A"
        }

        print("\nExtracted Animal Data:")
        for key, value in animal_data.items():
            print(f"{key}: {value}")

        return animal_data

    except Exception as e:
        print(f"Error occurred: {e}")
        return None

    finally:
        if driver:
            driver.quit()