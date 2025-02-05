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

        # Wait for results and parse
        time.sleep(2)
        cells = driver.find_elements(By.TAG_NAME, "td")

        animal_data = {
            'Animal_ID': cells[39].text,
            'Cage_ID': cells[41].text,
            'Strain': cells[42].text,
            'DoB': cells[44].text,
            'Age': cells[45].text,
            'No_of_animals': cells[47].text,
            'Sire': cells[48].text,
            'Dam': cells[49].text,
            'Team': cells[50].text,
            'PPL': cells[51].text,
            'Protocol': cells[52].text,
            'Room': cells[53].text,
            'Status': cells[54].text,
            'Project_code': cells[55].text,
            'Responsible_User': cells[57].text
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


# Usage
if __name__ == '__main__':
    # Example usage with prompt for credentials
    animal_info = get_animal_info("1773019")

    # Or with provided credentials
    # animal_info = get_animal_info("1773019", username="your_username", password="your_password")